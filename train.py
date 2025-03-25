

from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dataset import DataLoad
# from torch_geometric.nn.models import LightGCN
from model import MVMLCL
from torch_sparse import SparseTensor, matmul
from sklearn.model_selection import train_test_split

from sanfm import SANFM

# import random
from tqdm import tqdm

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    config = ModelConfig()

    Graph_data_1, Graph_data_2, Graph_data_3, long_tail_API_id, common_tags, mashup_des, api_des, mashup_des_test, api_des_test, mashup_tag, api_tag = DataLoad(
        device)
    k = set(long_tail_API_id["API_id"].tolist())
    # ------------ma-------------------
    ma_num_edges = len(Graph_data_1.edge_index[1])

    ma_all_index = [i for i in range(ma_num_edges)]
    ma_train_index, ma_test_index = train_test_split(ma_all_index, test_size=0.2, random_state=config.myseed)
    ma_train_edge_index = Graph_data_1.edge_index[:, ma_train_index].to(device)
    ma_test_edge_index = Graph_data_1.edge_index[:, ma_test_index].to(device)

    ma_train_sparse_edge_index = SparseTensor(row=ma_train_edge_index[0], col=ma_train_edge_index[1], sparse_sizes=(
        Graph_data_1.num_users + Graph_data_1.num_items, Graph_data_1.num_users + Graph_data_1.num_items)).to(device)
    ma_test_sparse_edge_index = SparseTensor(row=ma_test_edge_index[0], col=ma_test_edge_index[1], sparse_sizes=(
        Graph_data_1.num_users + Graph_data_1.num_items, Graph_data_1.num_users + Graph_data_1.num_items)).to(device)

    ma_sparse_edge_index = SparseTensor(row=Graph_data_1.edge_index[0], col=Graph_data_1.edge_index[1], sparse_sizes=(
        Graph_data_1.num_users + Graph_data_1.num_items, Graph_data_1.num_users + Graph_data_1.num_items)).to(device)

    # -------------mt-------------------
    mt_num_edges = len(Graph_data_2.edge_index[1])

    mt_all_index = [i for i in range(mt_num_edges)]
    mt_train_index, mt_test_index = train_test_split(mt_all_index, test_size=0.2, random_state=config.myseed)
    mt_train_edge_index = Graph_data_2.edge_index[:, mt_train_index].to(device)
    mt_test_edge_index = Graph_data_2.edge_index[:, mt_test_index].to(device)

    mt_train_sparse_edge_index = SparseTensor(row=mt_train_edge_index[0], col=mt_train_edge_index[1], sparse_sizes=(
        Graph_data_2.num_users + Graph_data_2.num_items, Graph_data_2.num_users + Graph_data_2.num_items)).to(device)
    mt_test_sparse_edge_index = SparseTensor(row=mt_test_edge_index[0], col=mt_test_edge_index[1], sparse_sizes=(
        Graph_data_2.num_users + Graph_data_2.num_items, Graph_data_2.num_users + Graph_data_2.num_items)).to(device)

    # --------------at----------------------
    at_num_edges = len(Graph_data_3.edge_index[1])

    at_all_index = [i for i in range(at_num_edges)]
    at_train_index, at_test_index = train_test_split(at_all_index, test_size=0.2, random_state=config.myseed)
    at_train_edge_index = Graph_data_3.edge_index[:, at_train_index].to(device)
    at_test_edge_index = Graph_data_3.edge_index[:, at_test_index].to(device)

    at_train_sparse_edge_index = SparseTensor(row=at_train_edge_index[0], col=at_train_edge_index[1], sparse_sizes=(
        Graph_data_3.num_users + Graph_data_3.num_items, Graph_data_3.num_users + Graph_data_3.num_items)).to(device)
    at_test_sparse_edge_index = SparseTensor(row=at_test_edge_index[0], col=at_test_edge_index[1], sparse_sizes=(
        Graph_data_3.num_users + Graph_data_3.num_items, Graph_data_3.num_users + Graph_data_3.num_items)).to(device)

    # -----------初始化模型-----------------
    model = MVMLCL(Graph_data_1=Graph_data_1, Graph_data_2=Graph_data_2, Graph_data_3=Graph_data_3,
                   Text_data_1=mashup_des,
                   Text_data_2=api_des, num_layers=config.num_layers,
                   add_self_loops=False, ma_sparse_edge_index=ma_sparse_edge_index)

    model = model.to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    optimizer = torch.optim.Adam(params=chain(model.parameters()), lr=config.lr,
                                 weight_decay=config.weight_decay)

    train_losses = []
    val_losses = []
    for epoch in tqdm(range(1, config.n_epoch + 1)):

        model.train()

        optimizer.zero_grad()

        # -------------------- 首先获取索引---------------------------------------
        user_indices1, pos_item_indices1, neg_item_indices1 = sample_mini_batch(config.train_batch_size,
                                                                                ma_train_edge_index)
        user_indices2, pos_item_indices2, neg_item_indices2 = sample_mini_batch(config.train_batch_size,
                                                                                mt_train_edge_index)
        user_indices3, pos_item_indices3, neg_item_indices3 = sample_mini_batch(config.train_batch_size,
                                                                                at_train_edge_index)

        # ------------------完成Mashup和API相似tag的索引查询
        # #这里应该是对tag进行一个判断，从而完成相同Tag的一个对比
        # #
        # #获取mashup_tag_id的列表
        # mashup_tag_list=mt_train_edge_index[1,pos_item_indices2].tolist()
        # #获取api_tag_id的列表
        # api_tag_list=at_train_edge_index[1,pos_item_indices3].tolist()
        # #common_tag中包含两列，mashup_tag_id,api_tag_id，该文件中是指mashup和api的tag相同的id的一个组合对
        # common_tags = pd.read_csv("./data/Common_tag.csv", sep='\t', header=0)
        # #mashup_tag_id的列表和api_tag_id的列表成对拼接，是指n*m的拼接，得到拼接数组k
        # #使用拼接数组k和common_tag做一个求取交集的过程得到m
        # # 将m作为成对的mashup和api进行返回
        # # 从common_tags中找出同时出现在mashup_tag_list和api_tag_list中的组合对
        # matching_pairs = common_tags[
        #     (common_tags["Mashup_tag_id"].isin(mashup_tag_list)) & (common_tags["API_tag_id"].isin(api_tag_list))]

        # --------tag_contrast
        # 获取mashup_tag_id的列表

        mashup_tag_list = mt_train_edge_index[1, pos_item_indices2].tolist()

        # 获取api_tag_id的列表
        api_tag_list = at_train_edge_index[1, pos_item_indices3].tolist()
        # 找出在common_tags中同时出现在mashup_tag_list和api_tag_list中的tag id ||| common_tags 读取包含mashup和api的tag相同的id组合对的数据文件
        matching_tags = common_tags[
            common_tags["Mashup_tag_id"].isin(mashup_tag_list) & common_tags["API_tag_id"].isin(api_tag_list)]

        # 获取匹配的mashup和api的tag id在原始pos_item_indices2和pos_item_indices3中所对应的值
        # 使用布尔索引找到匹配的索引
        matching_mashup_index = pos_item_indices2[np.isin(mashup_tag_list, matching_tags["Mashup_tag_id"])]
        matching_api_index = pos_item_indices3[np.isin(api_tag_list, matching_tags["API_tag_id"])]



        # -------------模型输入得出输出-------------------------------------------

        users_emb_final1, users_emb1_0, items_emb_final1, items_emb1_0, \
            users_emb_final2, users_emb2_0, items_emb_final2, items_emb2_0, \
            users_emb_final3, users_emb3_0, items_emb_final3, items_emb3_0, \
            mashup_des_pooled_output, api_des_pooled_output = model(
            ma_train_sparse_edge_index,
            mt_train_sparse_edge_index,
            at_train_sparse_edge_index, user_indices1, pos_item_indices1)

        # -----------ma_batch---------------------

        user_indices1 = user_indices1.to(device)
        pos_item_indices1 = pos_item_indices1.to(device)
        neg_item_indices1 = neg_item_indices1.to(device)

        users_emb_finalb1 = users_emb_final1[user_indices1]
        users_embb1_0 = users_emb1_0[user_indices1]
        pos_items_emb_final1 = items_emb_final1[pos_item_indices1]
        pos_items_emb1_0 = items_emb1_0[pos_item_indices1]
        neg_items_emb_final1 = items_emb_final1[neg_item_indices1]
        neg_items_emb1_0 = items_emb1_0[neg_item_indices1]
        # --------------mt_batch--------------------------

        user_indices2 = user_indices2.to(device)
        pos_item_indices2 = pos_item_indices2.to(device)
        neg_item_indices2 = neg_item_indices2.to(device)

        users_emb_finalb2 = users_emb_final2[user_indices2]
        users_embb2_0 = users_emb2_0[user_indices2]
        pos_items_emb_final2 = items_emb_final2[pos_item_indices2]
        pos_items_emb2_0 = items_emb2_0[pos_item_indices2]
        neg_items_emb_final2 = items_emb_final2[neg_item_indices2]
        neg_items_emb2_0 = items_emb2_0[neg_item_indices2]
        # --------------------at-batch-------------------

        user_indices3 = user_indices3.to(device)
        pos_item_indices3 = pos_item_indices3.to(device)
        neg_item_indices3 = neg_item_indices3.to(device)

        users_emb_finalb3 = users_emb_final3[user_indices3]
        users_embb3_0 = users_emb3_0[user_indices3]
        pos_items_emb_final3 = items_emb_final3[pos_item_indices3]
        pos_items_emb3_0 = items_emb3_0[pos_item_indices3]
        neg_items_emb_final3 = items_emb_final3[neg_item_indices3]
        neg_items_emb3_0 = items_emb3_0[neg_item_indices3]

        # ----------------BPR----------------------------

        train_loss_ma = bpr_loss(users_emb_finalb1, users_embb1_0, pos_items_emb_final1, pos_items_emb1_0,
                                 neg_items_emb_final1, neg_items_emb1_0, config.lamda)

        train_loss_mt = bpr_loss(users_emb_finalb2, users_embb2_0, pos_items_emb_final2, pos_items_emb2_0,
                                 neg_items_emb_final2, neg_items_emb2_0, config.lamda)

        train_loss_at = bpr_loss(users_emb_finalb3, users_embb3_0, pos_items_emb_final3, pos_items_emb3_0,
                                 neg_items_emb_final3, neg_items_emb3_0, config.lamda)

        # ----------------CL----------------------------

        # ----------------Local---------------------------
        M_MA_Graph = users_emb_final1[user_indices1]
        M_MT_Graph = users_emb_final2[user_indices1]
        # 用于判断索引在实际嵌入中的正确性
        # device = ma_train_edge_index.device
        # user_indices1 = user_indices1.to(device)
        #
        # my_test_set1 = ma_train_edge_index[0, user_indices1].tolist()
        #
        # device = mt_train_edge_index.device
        #
        # my_test_set2 = mt_train_edge_index[0, user_indices1].tolist()

        # 这里应该设置一个判断是否是长尾的判断，是的话就进行一个对比，不是则不进行对比，具体应该是在pos_index中进行判断。

        # 这一步是针对长尾部分进行挑选，从而使其进对比长尾部分的api

        # -------
        A_MA_Graph = items_emb_final1[pos_item_indices1]
        A_AT_Graph = users_emb_final3[pos_item_indices1]
        # 用于判断索引在实际嵌入中的正确性
        # device2 = ma_train_edge_index.device
        # pos_item_indices1 = user_indices1.to(device2)
        #
        # my_test_set3 = ma_train_edge_index[1, pos_item_indices1].tolist()
        #
        # my_test_set4 = at_train_edge_index[0, pos_item_indices1].tolist()

        # mashup_cl_local=CL_loss(users_emb_final1,users_emb_final2,config.temperature,b_cos=True)
        # api_cl_local = CL_loss(users_emb_final1, users_emb_final2, config.temperature, b_cos=True)

        T_M_Graph = items_emb_final2[matching_mashup_index]
        T_A_Graph = items_emb_final3[matching_api_index]

        # 相同是1 不同是0
        mask = get_masks_optimized_2(ma_train_edge_index, k, pos_item_indices1)

        cl_api_loss = contrastive_loss(A_MA_Graph, A_AT_Graph, mask, temperature=config.api_temperature)

        mashup_cl_local = CL_loss(M_MA_Graph, M_MT_Graph, config.temperature_local, b_cos=True)

        if matching_mashup_index.numel() > 1 and matching_api_index.numel() > 1:
            tag_cl_local = CL_loss(T_M_Graph, T_A_Graph, config.temperature_local, b_cos=True)
        else:
            tag_cl_local = 0
        # -----------------------------------------------------
        # ------------------Global------------------------------

        # -------------------首先将图结构中的嵌入进行拼接------------

        # -------------------------------------------------------

        mashup_cl_global = CL_loss(0.5 * (M_MA_Graph + M_MT_Graph), mashup_des_pooled_output, config.temperature_global,
                                   b_cos=True)
        api_cl_global = contrastive_loss(0.5 * (A_MA_Graph + A_AT_Graph), api_des_pooled_output, mask,
                                         temperature=config.api_temperature)

        # --------------------------------------
        #
        # Total_loss = train_loss_ma + train_loss_mt + train_loss_at + 0.01 * (
        #         mashup_cl_local + api_cl_local + tag_cl_local) + 0.01 * (
        #                      mashup_cl_global + api_cl_global )
        Total_loss = train_loss_ma + train_loss_mt + train_loss_at + 0.01 * (
                    mashup_cl_local + cl_api_loss + tag_cl_local) + 0.001 * (mashup_cl_global + api_cl_global)
        # Total_loss = train_loss_ma + train_loss_mt + train_loss_at + 0.01 * (
        #         mashup_cl_global + api_cl_global)
        Total_loss.backward()
        optimizer.step()

        # train_index = ma_train_edge_index[0, user_indices1].tolist()
        # temp_index = set(ma_test_edge_index[0].tolist())
        #
        # test_index_value = [user_indices1[index] for index, value in enumerate(train_index) if value in temp_index]
        # test_index_index = [index for index, value in enumerate(train_index) if value in temp_index]

        # mashup_sanfm_des_input = mashup_des_pooled_output[test_index_index] + M_MA_Graph[test_index_index]

        if (epoch % config.eval_steps == 0):
            model.eval()
            val_loss, recall, precision, ndcg,f1,map= my_evaluation(model, ma_test_edge_index, ma_test_sparse_edge_index,
                                                              [ma_train_edge_index], mt_test_edge_index,
                                                              mt_test_sparse_edge_index, [mt_train_edge_index],
                                                              at_test_edge_index, at_test_sparse_edge_index,
                                                              [at_train_edge_index], config.K, config.lamda,
                                                              user_indices1,
                                                              pos_item_indices1,
                                                              mashup_des_test, api_des_test, epoch)
            print(
                f"[Iteraion {epoch:05d}/{config.n_epoch}] train_loss: {train_loss_ma.item():.5f}, val_loss: {val_loss:.5f}, val_recall@{config.K}: {recall:.4f}, val_precision@{config.K}: {precision:.4f}, val_ndcg@{config.K}: {ndcg:.4f},val_f1@{config.K}: {f1:.5f},val_map@{config.K}: {map:.5f}")  #

