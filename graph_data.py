from itertools import product
import torch
from torch_geometric.data import Data
 

def create_torch_graph_data(sudoku_state, cursor_pos):
    
    # Define edge index
    row_edges = [[i*9+j, i*9+j+k] for i,j,k in product(range(9), range(9), range(-1,2)) if k!=0 and j+k>=0 and j+k<9]
    col_edges = [[i*9+j, (i+k)*9+j] for i,j,k in product(range(9), range(9), range(-1,2)) if k!=0 and i+k>=0 and i+k<9]
    square_edges = [[i*9+j, (i//3*3+k)*9+j//3*3+l] for i,j,k,l in product(range(9), range(9), range(3), range(3)) if k!=1 or l!=1]
    edge_index = torch.tensor(row_edges+col_edges+square_edges, dtype=torch.long).t().contiguous()

    # Define node features
    node_feature = torch.zeros((81, 10))  # One-hot encoding for cell values (0 for empty cells)
    node_feature[range(81), sudoku_state] = 1
    node_feature[cursor_pos, -1] = 1  # Add one-hot encoding for cursor position

    data = Data(x=node_feature, edge_index=edge_index)

    return data

def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

def save_model(model, path='default.pth'):
        torch.save(model.state_dict(), path)