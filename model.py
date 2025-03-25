import torch
import torch.nn as nn
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import LGConv
from torch_geometric.typing import Adj
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing

from utils import ModelConfig

config=ModelConfig()

class MVMLCL(MessagePassing):
    # G1 Mashup-API
    # G2 Mashup-tag
    # G3 API-tag
    # T1 Mashup
    # T2 API
    # T3 MT
    # T4 AT

    # def __init__(self, Graph_data_1, Graph_data_2, Graph_data_3, Text_data_1, Text_data_2, Text_data_3, Text_data_4,
    #              text_config, num_layers=3, add_self_loops=False):
    def __init__(self, Graph_data_1, Graph_data_2, Graph_data_3, Text_data_1, Text_data_2,
                 num_layers=3, add_self_loops=False,ma_sparse_edge_index=None):
        super().__init__()
        self.ma_sparse_edge_index = ma_sparse_edge_index
        self.num_layers = num_layers
        self.add_self_loops = add_self_loops
        self.alpha = torch.tensor([1 / (num_layers)] * (num_layers + 1))

        # ----------Mashup-API-----------------------
        self.num_users1 = Graph_data_1.num_users
        self.num_items1 = Graph_data_1.num_items
        self.embedding_dim1 = Graph_data_1.embedding_dim
        self.users_emb1 = nn.Embedding(num_embeddings=self.num_users1, embedding_dim=self.embedding_dim1)  # e_u^0
        self.items_emb1 = nn.Embedding(num_embeddings=self.num_items1, embedding_dim=self.embedding_dim1)  # e_i^0
        nn.init.normal_(self.users_emb1.weight, std=0.1)
        nn.init.normal_(self.items_emb1.weight, std=0.1)
        # ------------Mashup-Tag-----------------------
        self.num_users2 = Graph_data_2.num_users
        self.num_items2 = Graph_data_2.num_items
        self.embedding_dim2 = Graph_data_2.embedding_dim
        self.users_emb2 = nn.Embedding(num_embeddings=self.num_users2, embedding_dim=self.embedding_dim2)  # e_u^0
        self.items_emb2 = nn.Embedding(num_embeddings=self.num_items2, embedding_dim=self.embedding_dim2)  # e_i^0
        nn.init.normal_(self.users_emb2.weight, std=0.1)
        nn.init.normal_(self.items_emb2.weight, std=0.1)
        # -------------API-Tag---------------------------
        self.num_users3 = Graph_data_3.num_users
        self.num_items3 = Graph_data_3.num_items
        self.embedding_dim3 = Graph_data_3.embedding_dim
        self.users_emb3 = nn.Embedding(num_embeddings=self.num_users3, embedding_dim=self.embedding_dim3)  # e_u^0
        self.items_emb3 = nn.Embedding(num_embeddings=self.num_items3, embedding_dim=self.embedding_dim3)  # e_i^0
        nn.init.normal_(self.users_emb3.weight, std=0.1)
        nn.init.normal_(self.items_emb3.weight, std=0.1)
        # -----------------------------------------------

        self.T1 = Text_data_1
        self.T2 = Text_data_2
        self.relu = nn.ELU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        # self.input_dim = text_config.input_dim
        # self.hidden_dim = text_config.hidden_dim
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.mashup_fc = nn.Linear(self.input_dim, self.hidden_dim)
        self.api_fc = nn.Linear(self.input_dim, self.hidden_dim)
        self.tag_fc = nn.Linear(self.input_dim, self.hidden_dim)
        self.api_layers = nn.Sequential(
            self.api_fc,
            self.relu,
            self.dropout
        )
        self.mashup_layers = nn.Sequential(

            self.mashup_fc,
            self.relu,
            self.dropout
        )

        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])
        self.reset_parameters()

    def reset_parameters(self):
        # -----------------------MA------------------------
        torch.nn.init.xavier_uniform_(self.users_emb1.weight)
        torch.nn.init.xavier_uniform_(self.items_emb1.weight)
        # -----------------------MT------------------------
        torch.nn.init.xavier_uniform_(self.users_emb2.weight)
        torch.nn.init.xavier_uniform_(self.items_emb2.weight)
        # ----------------------AT---------------------------
        torch.nn.init.xavier_uniform_(self.users_emb3.weight)
        torch.nn.init.xavier_uniform_(self.items_emb3.weight)

        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, edge_index1: SparseTensor, edge_index2: SparseTensor, edge_index3: SparseTensor,index_m,index_a):
        """Forward propagation of LightGCN Model.

        Args:
            edge_index (SparseTensor): adjacency matrix

        Returns:
            tuple (Tensor): e_u_k, e_u_0, e_i_k, e_i_0
            :param index_a:
            :param index_m:
            :param edge_index1:
            :param edge_index3:
            :param edge_index2:
        """

        # -------------MA-------------------
        edge_index_norm1 = edge_index1
        x = torch.cat([self.users_emb1.weight, self.items_emb1.weight])

        out = self.get_embedding(edge_index_norm1, x)

        users_emb_final1, items_emb_final1 = torch.split(out, [self.num_users1,
                                                               self.num_items1])  # splits into e_u^K and e_i^K

        # -------------MT-------------------
        edge_index_norm2 = edge_index2
        x = torch.cat([self.users_emb2.weight, self.items_emb2.weight])

        out = self.get_embedding(edge_index_norm2, x)

        users_emb_final2, items_emb_final2 = torch.split(out, [self.num_users2,
                                                               self.num_items2])  # splits into e_u^K and e_i^K
        # -------------AT-------------------
        edge_index_norm3 = edge_index3
        x = torch.cat([self.users_emb3.weight, self.items_emb3.weight])

        out = self.get_embedding(edge_index_norm3, x)

        users_emb_final3, items_emb_final3 = torch.split(out, [self.num_users3,
                                                               self.num_items3])  # splits into e_u^K and e_i^K
        # -------------------------------


        # ------------M----------------------
        mashup_des=self.T1[index_m]
        api_des=self.T2[index_a]



        #------------------------------------
        mashup_des_pooled_output = self.process_input(mashup_des, "mashup")
        api_des_pooled_output = self.process_input(api_des, "api")


        return users_emb_final1, self.users_emb1.weight, items_emb_final1, self.items_emb1.weight, \
            users_emb_final2, self.users_emb2.weight, items_emb_final2, self.items_emb2.weight, \
            users_emb_final3, self.users_emb3.weight, items_emb_final3, self.items_emb3.weight, \
            mashup_des_pooled_output, api_des_pooled_output
    def process_input(self, emb, type=None):
        processed_output=None
        if type == "api":
            processed_output = self.api_layers(emb)
        elif type == "mashup":
            processed_output = self.mashup_layers(emb)
        return processed_output


    def get_final_emb(self):
        x = torch.cat([self.users_emb1.weight, self.items_emb1.weight])
        out = x * self.alpha[0]
        # print(f"{out.shape=}")
        # print(f"{x.shape=}")

        for i in range(self.num_layers):
            x = self.convs[i](x, self.ma_sparse_edge_index)
            # print(f"{x.shape=}")
            out = out + x * self.alpha[i + 1]
            # print(f"{out.shape=}")
            # print(f"{x.shape=}")

        users_emb_final1, items_emb_final1 = torch.split(out, [self.num_users1,
                                                               self.num_items1])  # splits into e_u^K and e_i^K

        # returns e_u^K, e_u^0, e_i^K, e_i^0
        return users_emb_final1, items_emb_final1
