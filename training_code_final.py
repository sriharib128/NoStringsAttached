# %%
# whether you are using a GPU to run this Colab
use_gpu = False
# whether you are using a custom GCE env to run the Colab (uses different CUDA)
custom_GCE_env = False

# %% [markdown]
# # Installation

# %%
import os
import torch
import torch_geometric
import numpy as np
import math
torch_geometric.__version__

# %%
class FB15kDataset(torch_geometric.data.InMemoryDataset):
  r"""FB15-237 dataset from Freebase.
  Follows similar structure to torch_geometric.datasets.rel_link_pred_dataset

  Args:
    root (string): Root directory where the dataset should be saved.
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
  """
  data_path = 'https://raw.githubusercontent.com/DeepGraphLearning/' \
              'KnowledgeGraphEmbedding/master/data/FB15k-237'
  def __init__(self, root, transform=None, pre_transform=None):
    super().__init__(root, transform, pre_transform)
    self.data, self.slices = torch.load(self.processed_paths[0])

  @property
  def raw_file_names(self):
    return ['train.txt', 'valid.txt', 'test.txt',
            'entities.dict', 'relations.dict']

  @property
  def processed_file_names(self):
    return ['data.pt']

  @property
  def raw_dir(self):
    return os.path.join(self.root, 'raw')

  def download(self):
      for file_name in self.raw_file_names:
        torch_geometric.data.download_url(f'{self.data_path}/{file_name}',
                                          self.raw_dir)

  def process(self):
    with open(os.path.join(self.raw_dir, 'entities.dict'), 'r') as f:
      lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
      entities_dict = {key: int(value) for value, key in lines}

    with open(os.path.join(self.raw_dir, 'relations.dict'), 'r') as f:
      lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
      relations_dict = {key: int(value) for value, key in lines}

    kwargs = {}
    for split in ['train', 'valid', 'test']:
      with open(os.path.join(self.raw_dir, f'{split}.txt'), 'r') as f:
        lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
        heads = [entities_dict[row[0]] for row in lines]
        relations = [relations_dict[row[1]] for row in lines]
        tails = [entities_dict[row[2]] for row in lines]
        kwargs[f'{split}_edge_index'] = torch.tensor([heads, tails])
        kwargs[f'{split}_edge_type'] = torch.tensor(relations)

    _data = torch_geometric.data.Data(num_entities=len(entities_dict),
                                      num_relations=len(relations_dict),
                                      **kwargs)

    if self.pre_transform is not None:
      _data = self.pre_transform(_data)

    data, slices = self.collate([_data])

    torch.save((data, slices), self.processed_paths[0])

FB15k_dset = FB15kDataset(root='FB15k')
data = FB15k_dset[0]

# %%
print(f'The graph has a total of {data.num_entities} entities and {data.num_relations} relations.')
print(f'The train split has {data.train_edge_type.size()[0]} relation triples.')
print(f'The valid split has {data.valid_edge_type.size()[0]} relation triples.')
print(f'The test split has {data.test_edge_type.size()[0]} relation triples.')

# %%
def load_dict(file_path):
    """
    Loads a dictionary file where each non-empty line is formatted as:
       <value>\t<id>
    and returns a dict mapping the id (string) to its integer value.
    """
    mapping = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            val, key = parts[0], parts[1]
            mapping[key] = int(val)
    return mapping
entities_dict_path = os.path.join('FB15k', 'raw', 'entities.dict')
entities_dict = load_dict(entities_dict_path)

# %%
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversal(torch.nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class TransE(torch.nn.Module):
    def __init__(self, num_entities, num_relations, device, embedding_dim=200,
                 margin=1.0, visualize=False, sensitive_indices=None, profession_indices=None,
                 lambda_eq=0.1, lambda_ortho=0.1, lambda_adv=0.1, profession_gender_labels=None):
        super(TransE, self).__init__()
        self.device = device
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.lambda_eq = lambda_eq
        self.lambda_ortho = lambda_ortho
        self.lambda_adv = lambda_adv
        self.sensitive_indices = sensitive_indices
        self.profession_indices = torch.tensor(profession_indices, device=self.device) if profession_indices is not None else None
        self.profession_gender_labels = profession_gender_labels.to(device) if profession_gender_labels is not None else None

        # Embeddings
        self.entities_emb = self.init_emb(num_entities, embedding_dim, emb_type='entity')
        self.relations_emb = self.init_emb(num_relations, embedding_dim, emb_type='relation')
        self.criterion = torch.nn.MarginRankingLoss(margin=margin, reduction='none')

        # Adversarial components
        self.grl = GradientReversal(alpha=1.0)
        self.adversary = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 1),
            torch.nn.Sigmoid()
        )

    def init_emb(self, size, dim, emb_type='relation'):
        emb = torch.nn.Embedding(num_embeddings=size, embedding_dim=dim)
        uniform_max = 6 / np.sqrt(dim)
        emb.weight.data.uniform_(-uniform_max, uniform_max)
        if emb_type == 'relation':
            emb_norm = torch.norm(emb.weight.data, dim=1, keepdim=True)
            emb.weight.data = emb.weight.data / emb_norm
        return emb

    def forward(self, edge_index, negative_edge_index, edge_type):
        positive_distance = self.distance(edge_index, edge_type)
        negative_distance = self.distance(negative_edge_index, edge_type)
        main_loss = self.loss(positive_distance, negative_distance)
        # print(edge_index,"======\n" ,negative_edge_index,"======\n", edge_type,"======\n\n\n\n")

        if self.sensitive_indices is not None:
            print(edge_index[0, :])
            # Convert tensor indices to integers
            head_indices = edge_index[0, :].cpu().numpy()  # Handles GPU tensors
            names = [entities_dict[idx] for idx in head_indices]
            print(names)
            print("="*10)
            print(edge_index[1,:])
            print(entities_dict[edge_index[0,:].item()])

            current_entities = torch.unique(edge_index)
            if self.profession_indices is not None:
                mask = torch.isin(current_entities, self.profession_indices)
                entities_to_consider = current_entities[mask]
            else:
                entities_to_consider = current_entities
            eq_loss, ortho_loss = self.debias_loss(entities_to_consider)
            total_loss = main_loss + self.lambda_eq * eq_loss + self.lambda_ortho * ortho_loss

            # Adversarial loss
            if self.profession_gender_labels is not None and self.lambda_adv != 0 and len(entities_to_consider) > 0:
                adv_loss = self.adversarial_loss(entities_to_consider)
                total_loss += self.lambda_adv * adv_loss
                if(eq_loss !=0):
                    print("main Loss = ", main_loss.item()," eq_loss = ", eq_loss.item(), "ortho_loss = ", ortho_loss.item()," adv_loss = ", adv_loss.item()," total_loss = ", total_loss.item())
            else:
                if(eq_loss !=0):
                    print("main Loss = ", main_loss.item()," eq_loss = ", eq_loss.item(), "ortho_loss = ", ortho_loss.item()," total_loss = ", total_loss.item())
            return total_loss
        else:
            return main_loss

    def adversarial_loss(self, current_profession_entities):
        profession_embs = self.entities_emb(current_profession_entities)
        reversed_embs = self.grl(profession_embs)
        preds = self.adversary(reversed_embs).squeeze()

        # Find positions in profession_indices
        mask = (current_profession_entities.unsqueeze(1) == self.profession_indices.unsqueeze(0))
        pos_in_profession = torch.argmax(mask.int(), dim=1)
        labels = self.profession_gender_labels[pos_in_profession]

        adv_loss = torch.nn.functional.binary_cross_entropy(preds, labels.float())
        return adv_loss

    # The rest of the methods (distance, loss, debias_loss, predict) remain unchanged as in the original code
    def predict(self, edge_index, edge_type):
        return self.distance(edge_index, edge_type)

    def distance(self, edge_index, edge_type):
        heads = edge_index[0, :]
        tails = edge_index[1, :]
        return (self.entities_emb(heads) + self.relations_emb(edge_type) -
                self.entities_emb(tails)).norm(p=2., dim=1, keepdim=True)

    def loss(self, positive_distance, negative_distance):
        y = torch.tensor([-1], dtype=torch.long, device=self.device).expand_as(positive_distance)
        return self.criterion(positive_distance, negative_distance, y).sum()

    def debias_loss(self, current_entities):
        if isinstance(self.sensitive_indices, list):
            sensitive_idx_tensor = torch.tensor(self.sensitive_indices, device=self.device)
        else:
            sensitive_idx_tensor = self.sensitive_indices.to(self.device)
        
        mask = torch.isin(current_entities, sensitive_idx_tensor)
        if mask.sum() == 0 or mask.sum() == current_entities.size(0):
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
        
        sensitive_batch = current_entities[mask]
        non_sensitive_batch = current_entities[~mask]

        sensitive_emb = self.entities_emb(sensitive_batch)
        non_sensitive_emb = self.entities_emb(non_sensitive_batch)

        distances = torch.cdist(sensitive_emb, non_sensitive_emb, p=2)
        mean_dist = torch.mean(distances, dim=1, keepdim=True)
        eq_loss = torch.mean((distances - mean_dist) ** 2)

        dot_products = torch.matmul(sensitive_emb, non_sensitive_emb.transpose(0, 1))
        ortho_loss = torch.mean(dot_products ** 2)

        return eq_loss, ortho_loss

# %%


# %% [markdown]
# # Training and Testing
# 
# Now that we have implemented data processing and the model, it is time to train and test on the task of predicting missing tails.

# %% [markdown]
# One key aspect of training our model is creating corrupted triples by either replacing the head or tail with a random entity. We will do this once for every epoch, randomly replacing heads and tails

# %%
# def create_corrupted_edge_index(edge_index, edge_type, num_entities):
#   corrupt_head_or_tail = torch.randint(high=2, size=edge_type.size(),
#                                        device=device)
#   random_entities = torch.randint(high=num_entities,
#                                   size=edge_type.size(), device=device)
#   # corrupt when 1, otherwise regular head
#   heads = torch.where(corrupt_head_or_tail == 1, random_entities,
#                       edge_index[0,:])
#   # corrupt when 0, otherwise regular tail
#   tails = torch.where(corrupt_head_or_tail == 0, random_entities,
#                       edge_index[1,:])
#   return torch.stack([heads, tails], dim=0)
def create_corrupted_edge_index(edge_index, edge_type, num_entities, negative_ratio=1):
    corrupted_indices = []
    for _ in range(negative_ratio):
        corrupt_head_or_tail = torch.randint(high=2, size=edge_type.size(), device=device)
        random_entities = torch.randint(high=num_entities, size=edge_type.size(), device=device)
        heads = torch.where(corrupt_head_or_tail == 1, random_entities, edge_index[0,:])
        tails = torch.where(corrupt_head_or_tail == 0, random_entities, edge_index[1,:])
        corrupted_indices.append(torch.stack([heads, tails], dim=0))
    return torch.cat(corrupted_indices, dim=1)

# %% [markdown]
# Other than corrupting samples, the training process is pretty standard. One thing to keep in mind is training samples are shuffled between epochs.

# %%
def train(model, data, optimizer, device, epochs=50, batch_size=128,
          eval_batch_size=256, valid_freq=5):
  train_edge_index = data.train_edge_index.to(device)
  train_edge_type = data.train_edge_type.to(device)

  best_valid_score = 0
  valid_scores = None
  test_scores = None
  for epoch in range(epochs):
    model.train()
    # e = e / ||e||
    # entities_norm = torch.norm(model.entities_emb.weight.data, dim=1, keepdim=True)
    # model.entities_emb.weight.data = model.entities_emb.weight.data / entities_norm

    # shuffle train set each batch
    num_triples = data.train_edge_type.size()[0]
    shuffled_triples_order = np.arange(num_triples)
    np.random.shuffle(shuffled_triples_order)
    shuffled_edge_index = train_edge_index[:, shuffled_triples_order]
    shuffled_edge_type = train_edge_type[shuffled_triples_order]

    negative_edge_index = create_corrupted_edge_index(shuffled_edge_index,
                                                      shuffled_edge_type,
                                                      data.num_entities)

    total_loss = 0
    total_size = 0
    for batch_idx in range(math.ceil(num_triples / batch_size)):
      # Do this at the batch level, not just epoch level
      entities_norm = torch.norm(model.entities_emb.weight.data, dim=1, keepdim=True)
      model.entities_emb.weight.data = model.entities_emb.weight.data / entities_norm
      batch_start = batch_idx * batch_size
      batch_end = (batch_idx + 1) * batch_size
      batch_edge_index = shuffled_edge_index[:,batch_start:batch_end]
      batch_negative_edge_index = negative_edge_index[:,batch_start:batch_end]
      batch_edge_type = shuffled_edge_type[batch_start:batch_end]
      loss = model(batch_edge_index, batch_negative_edge_index, batch_edge_type)
      total_loss += loss.item()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_size += batch_edge_type.size()[0]
      break
      # if model.visualize  and epoch == 0 \
      # and batch_idx % 100 == 0:
      #   visualize_emb(model, batch_idx)
    break
    
    print(f'Epoch {epoch}, train loss equals {total_loss / total_size}')
    if (epoch + 1) % valid_freq == 0:
      mrr_score, mr_score, hits_at_10 = eval(model, data.valid_edge_index.to(device),
                                             data.valid_edge_type.to(device),
                                             data.num_entities, device)
      print(f'Validation score equals {mrr_score}, {mr_score}, {hits_at_10}')
      if mrr_score > best_valid_score:
        valid_scores = (mrr_score, mr_score, hits_at_10)
        test_mmr_score, test_mr_score, test_hits_at_10 = \
                                        eval(model, data.valid_edge_index.to(device),
                                             data.valid_edge_type.to(device),
                                             data.num_entities, device)
        test_scores = (test_mmr_score, test_mr_score, test_hits_at_10)
  print(f'Test scores from best model (mmr, mr, h@10): {test_scores}')

# %%
def mrr(predictions, gt):
  indices = predictions.argsort()
  return (1.0 / (indices == gt).nonzero()[:, 1].float().add(1.0)).sum().item()

def mr(predictions, gt):
  indices = predictions.argsort()
  return ((indices == gt).nonzero()[:, 1].float().add(1.0)).sum().item()

def hit_at_k(predictions, gt, device, k=10):
  zero_tensor = torch.tensor([0], device=device)
  one_tensor = torch.tensor([1], device=device)
  _, indices = predictions.topk(k=k, largest=False)
  return torch.where(indices == gt, one_tensor, zero_tensor).sum().item()

# %%
def eval(model, edge_index, edge_type, num_entities, device, eval_batch_size=64):
  model.eval()
  num_triples = edge_type.size()[0]
  mrr_score = 0
  mr_score = 0
  hits_at_10 = 0
  num_predictions = 0

  for batch_idx in range(math.ceil(num_triples / eval_batch_size)):
    batch_start = batch_idx * eval_batch_size
    batch_end = (batch_idx + 1) * eval_batch_size
    batch_edge_index = edge_index[:,batch_start:batch_end]
    batch_edge_type = edge_type[batch_start:batch_end]
    batch_size = batch_edge_type.size()[0] # can be different on last batch

    all_entities = torch.arange(end=num_entities,
                                device=device).unsqueeze(0).repeat(batch_size, 1)
    head_repeated = batch_edge_index[0,:].reshape(-1, 1).repeat(1, num_entities)
    relation_repeated = batch_edge_type.reshape(-1, 1).repeat(1, num_entities)

    head_squeezed = head_repeated.reshape(-1)
    relation_squeezed = relation_repeated.reshape(-1)
    all_entities_squeezed = all_entities.reshape(-1)

    entity_index_replaced_tail = torch.stack((head_squeezed,all_entities_squeezed))
    predictions = model.predict(entity_index_replaced_tail, relation_squeezed)
    predictions = predictions.reshape(batch_size, -1)
    gt = batch_edge_index[1,:].reshape(-1, 1)

    mrr_score += mrr(predictions, gt)
    mr_score += mr(predictions, gt)
    hits_at_10 += hit_at_k(predictions, gt, device=device, k=10)
    num_predictions += predictions.size()[0]

  mrr_score = mrr_score / num_predictions
  mr_score = mr_score / num_predictions
  hits_at_10 = hits_at_10 / num_predictions
  return mrr_score, mr_score, hits_at_10

# %%
import pandas as pd

# Load the file
file_path = "gen2prof_fair_all.txt"

# Read the file and extract gender-profession pairs
gender_prof_pairs = []
with open(file_path, "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 5:
            gender_id, human_id, profession_id = parts[0], parts[2], parts[4]
            gender_prof_pairs.append((gender_id, human_id ,profession_id))

# Convert to DataFrame
df = pd.DataFrame(gender_prof_pairs, columns=["Gender_ID","Human_ID", "Profession_ID"])

# Get top 10 most common professions
profession_entitiy_ids = np.unique(df["Profession_ID"].values).tolist()
len(profession_entitiy_ids)

# %%
def load_dict(file_path):
    """
    Loads a dictionary file where each non-empty line is formatted as:
       <value>\t<id>
    and returns a dict mapping the id (string) to its integer value.
    """
    mapping = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            val, key = parts[0], parts[1]
            mapping[key] = int(val)
    return mapping

lr = 0.01
if use_gpu:
  epochs = 50
  valid_freq = 5
else:
  epochs = 50
  valid_freq = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
male_entity_id = '/m/05zppz'
female_entity_id = '/m/02zsn'
entities_dict_path = os.path.join('FB15k', 'raw', 'entities.dict')
relations_dict_path = os.path.join('FB15k', 'raw', 'relations.dict')

entities_dict = load_dict(entities_dict_path)
relations_dict = load_dict(relations_dict_path)

male_idx = entities_dict[male_entity_id]
female_idx = entities_dict[female_entity_id]

sensitive_indices = [male_idx,female_idx]
profession_indices = [entities_dict[id] for id in profession_entitiy_ids]
print (sensitive_indices,profession_indices)

# %%
model = TransE(data.num_entities, data.num_relations, device, visualize=False,
               sensitive_indices=sensitive_indices, profession_indices = profession_indices,lambda_eq=0.1, lambda_ortho=0.1).to(device)
    # model = TransE(data.num_entities, data.num_relations, device, visualize=False).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
train(model, data, optimizer, device, epochs=epochs, valid_freq=valid_freq)

# %%
import os
import torch
import numpy as np

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Helper Functions ---

def score_func(e, r, t):
    """
    Our scoring function for a triple.
    We define score = -||e + r - t||_2,
    so that a higher score indicates a better (more likely) triple.
    """
    return -torch.norm(e + r - t, p=2)

# def score_func(e, r, t):
#     """
#     Our scoring function for a triple.
#     We define score = cosine_similarity(e + r, t),
#     so that a higher score indicates a better (more likely) triple.
#     """
#     e_r = e + r
#     cos_sim = torch.nn.functional.cosine_similarity(e_r, t, dim=0)
#     return cos_sim

def load_dict(file_path):
    """
    Loads a dictionary file where each non-empty line is formatted as:
       <value>\t<id>
    and returns a dict mapping the id (string) to its integer value.
    """
    mapping = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            val, key = parts[0], parts[1]
            mapping[key] = int(val)
    return mapping

# --- Load FB15k Dictionaries ---
# Adjust these paths as needed.
entities_dict_path = os.path.join('FB15k', 'raw', 'entities.dict')
relations_dict_path = os.path.join('FB15k', 'raw', 'relations.dict')

entities_dict = load_dict(entities_dict_path)
relations_dict = load_dict(relations_dict_path)

# --- Define Sensitive Attributes and Relations ---
# In our case, we want to shift human entity embeddings to be “more male”.
# We assume:
#   male attribute: "/m/05zppz"
#   female attribute: "/m/02zsn"
#   sensitive relation: "/people/person/gender"
#   profession relation: "/people/person/profession"

male_entity_id = '/m/05zppz'
female_entity_id = '/m/02zsn'
gender_relation_id = '/people/person/gender'
profession_relation_id = '/people/person/profession'

# Get indices for these from the dictionaries
if male_entity_id not in entities_dict:
    raise ValueError(f"Male entity id {male_entity_id} not found in entities_dict.")
if female_entity_id not in entities_dict:
    raise ValueError(f"Female entity id {female_entity_id} not found in entities_dict.")
if gender_relation_id not in relations_dict:
    raise ValueError(f"Gender relation id {gender_relation_id} not found in relations_dict.")
if profession_relation_id not in relations_dict:
    raise ValueError(f"Profession relation id {profession_relation_id} not found in relations_dict.")

male_idx = entities_dict[male_entity_id]
female_idx = entities_dict[female_entity_id]
r_gender_idx = relations_dict[gender_relation_id]
r_prof_idx = relations_dict[profession_relation_id]

# --- Define Target Professions ---
target_professions = profession_entitiy_ids

# Map target profession ids to their entity indices.
target_prof_indices = {}
for prof in target_professions:
    if prof in entities_dict:
        target_prof_indices[prof] = entities_dict[prof]
    else:
        print(f"Warning: Target profession {prof} not found in entities_dict.")

# --- Load the gen2prof_fair_all.txt File ---
# The file format is:
#   gender_attr   /people/person/gender   human_entity   /people/person/profession   profession_entity
file_path = 'gen2prof_fair_all.txt'
with open(file_path, 'r') as f:
    lines = f.readlines()

# --- Set Hyperparameter for Bias Update ---
alpha = 0.1  # learning rate for the gradient update

# %%
# --- Accumulators for Bias Scores ---
# For each target profession, we accumulate the delta score and count occurrences.
bias_accumulator = {prof: 0.0 for prof in target_prof_indices.keys()}
count_accumulator = {prof: 0 for prof in target_prof_indices.keys()}

# --- Main Loop: Process Each Occurrence ---
for line in lines:
    tokens = line.strip().split('\t')
    # Expecting 5 tokens: [gender_attr, gender_relation, human_entity, profession_relation, profession_entity]
    if len(tokens) < 5:
        continue
    gender_token = tokens[0]       # e.g. '/m/02zsn' (female) or '/m/05zppz' (male) — not used directly here.
    human_id = tokens[2]           # e.g. '/m/010hn'
    prof_id = tokens[4]            # e.g. '/m/0n1h'
    
    # Only process if this occurrence is one of our target professions.
    if prof_id not in target_prof_indices:
        continue
    
    # Lookup human entity index; skip if not found.
    if human_id not in entities_dict:
        continue
    human_idx = entities_dict[human_id]

    # --- Retrieve and Prepare Embeddings ---
    # Get the original human embedding (e_j) from the trained model.
    # (Assumes model.entities_emb is a Parameter containing the entity embeddings.)
    e_j_orig = model.entities_emb.weight[human_idx].clone().detach().to(device)
    
    # Create a copy that requires gradient for the bias update.
    e_j = e_j_orig.clone().detach().requires_grad_(True)

    # Get the gender relation embedding.
    r_gender = model.relations_emb.weight[r_gender_idx].detach().to(device)
  
    # Get the male and female attribute embeddings.
    e_male = model.entities_emb.weight[male_idx].detach().to(device)
    e_female = model.entities_emb.weight[female_idx].detach().to(device)
    
    # --- Compute the Bias Score Function m(θ) ---
    # We define m(θ) = score(e_j, r_gender, e_male) - score(e_j, r_gender, e_female)
    score_male = score_func(e_j, r_gender, e_male)
    score_female = score_func(e_j, r_gender, e_female)
    m_val = score_male - score_female  # Higher m_val means the human is more aligned with male.
    
    # Compute gradient of m with respect to e_j.
    m_val.backward()
    
    # Perform the update: e_j' = e_j_orig + α * (∂m/∂e_j)
    new_e_j = e_j_orig + alpha * e_j.grad
    
    # --- Compute the Change in Profession Score ---
    # Get the profession relation embedding.
    r_prof = model.relations_emb.weight[r_prof_idx].detach().to(device)
    
    # Get the profession embedding
    e_p = model.entities_emb.weight[target_prof_indices[prof_id]].detach().to(device)

    # Compute the original and updated scores for (human, profession) relation
    score_before = score_func(e_j_orig, r_prof, e_p)
    score_after = score_func(new_e_j, r_prof, e_p)

    # Compute ∇ₚ = score_after - score_before
    delta_p = np.abs(score_after - score_before)

    # Accumulate bias score for this profession
    bias_accumulator[prof_id] += delta_p.item()
    count_accumulator[prof_id] += 1

# --- Compute Final Bias Scores ---
final_bias_scores = {}
for prof_id, total_bias in bias_accumulator.items():
    if count_accumulator[prof_id] > 0:
        final_bias_scores[prof_id] = total_bias / count_accumulator[prof_id]
    else:
        final_bias_scores[prof_id] = None  # No occurrences found

# --- Print Results ---
print("\nBias Scores for Target Professions:")
for prof_id, bias_score in final_bias_scores.items():
    if bias_score is not None:
        print(f"Profession {prof_id}: Bias Score = {bias_score:.8f}")
    else:
        print(f"Profession {prof_id}: No occurrences found in dataset")

# %%
import json

# Save the final_bias_scores dictionary to a JSON file
with open('final_bias_scores_debiasing.json', 'w') as file:
    json.dump(final_bias_scores, file, indent=4)

print("Bias scores have been saved to 'final_bias_scores.json'.")


# %%



