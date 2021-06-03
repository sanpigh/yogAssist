import itertools
import numpy as np
from yogAssist.utils import *

KPTS_TO_NEIGHBOURS_DEF = {  'nose':['sternum'],
                            'sternum':['nose','Rshoulder','Rhip','Lhip','Lshoulder'],
                            'Rshoulder':['Relbow','Rhip', 'sternum', 'Lshoulder'],
                            'Lshoulder':['Lelbow','Lhip', 'sternum', 'Rshoulder'],
                            'Relbow':['Rshoulder', 'Rwrist'],
                            'Lelbow':['Lshoulder', 'Lwrist'],
                            'Rwrist':['Relbow'],
                            'Lwrist':['Lelbow'],
                            'Rhip':['Lhip', 'sternum', 'Rshoulder', 'Rknee'],
                            'Lhip':['Rhip', 'sternum', 'Lshoulder', 'Lknee'],
                            'Rknee':['Rhip', 'Rankle'],
                            'Lknee':['Lhip', 'Lankle'],
                            'Rankle':['Rknee'],
                            'Lankle':['Lknee']}

KPTS_TO_NEIGHBOURS_DEF_API = {  'nose':['rightShoulder', 'leftShoulder'],
                                'rightShoulder':['rightElbow','leftShoulder', 'rightHip'],
                                'leftShoulder':['leftElbow','rightShoulder', 'leftHip'],
                                'rightElbow':['rightShoulder', 'rightWrist'],
                                'leftElbow':['leftShoulder', 'leftWrist'],
                                'rightWrist':['rightElbow'],
                                'leftWrist':['leftElbow'],
                                'rightHip':['leftHip', 'rightShoulder', 'rightKnee'],
                                'leftHip':['rightHip', 'leftShoulder', 'leftKnee'],
                                'rightKnee':['rightHip', 'rightAnkle'],
                                'leftKnee':['leftHip', 'leftAnkle'],
                                'rightAnkle':['rightKnee'],
                                'leftAnkle':['leftKnee']}
class Scoring:
    
    kpts_to_neighbor_dict_model_embedded = KPTS_TO_NEIGHBOURS_DEF
    kpts_to_neighbor_dict_js_api = KPTS_TO_NEIGHBOURS_DEF_API
        
    def __init__(self, keypoints, local=True):
        self.keypoints = keypoints
        self.kpts_to_neighbor_dict =  self.kpts_to_neighbor_dict_model_embedded if local else self.kpts_to_neighbor_dict_js_api
    
    def get_coordinates(self, node_name):
        coord = []
        if node_name in self.keypoints.keys():
            coord = self.keypoints[node_name] 
        return coord

    def compute_kpts_to_neighbor_from_ref_dictionnary(self):
        
        # create list of dictionnaries for predicted keypoints
        kpts_to_neighbor_predict = {}
        
        # iterate through predicted keypoints      
        for keypoint,coord in self.keypoints.items():
            # check keypoint is a candidate for neighbours
            if keypoint in self.kpts_to_neighbor_dict.keys():
                # get the candidate neighbours (at least 2 to compute angles and cosine simalarities)
                neighbours_d = [ neighbour for neighbour in self.kpts_to_neighbor_dict[keypoint] ]
                if len(neighbours_d) == 1:
                    continue    
                # create node dict
                node_dict = {}
                node_dict[keypoint] = coord
                # create last dict levels: keypoint name, keypoint coordinates
                neighbours_dict = {}
                for n in neighbours_d:
                    if self.get_coordinates(n):
                        neighbours_dict[n] = self.get_coordinates(n)
                node_dict['neighbours'] = neighbours_dict
                kpts_to_neighbor_predict[keypoint] = node_dict
                
        return kpts_to_neighbor_predict
                      
    def compute_cosine_similarities(self) -> dict:
        cosin_sim_dict = {}
        # create list of dictionnaries for predicted keypoints
        kpts_to_neighbor_predict = self.compute_kpts_to_neighbor_from_ref_dictionnary()
        
        for keypoint,keypoint_attr in kpts_to_neighbor_predict.items():
            
            number_of_neighbours_ = len(keypoint_attr['neighbours'])

            if number_of_neighbours_ < 2:
                continue  
            
            for idx in range(0, number_of_neighbours_):
                for idx_next in range(idx+1, number_of_neighbours_):
                    # create pairs dictionnary
                    cosine_sim_key_entry = f"{keypoint}-({list(keypoint_attr['neighbours'].keys())[idx]}, {list(keypoint_attr['neighbours'].keys())[idx_next]})"
                    # create A, B , C points
                    dot_A_vect = keypoint_attr[keypoint]
                    dot_B_vect = list(keypoint_attr['neighbours'].values())[idx]
                    dot_C_vect = list(keypoint_attr['neighbours'].values())[idx_next] 
                    # compute ((xB - xA), (yB, yA))     
                    pair_1_scalar_vectors = (dot_B_vect[0]-dot_A_vect[0], 
                                            dot_B_vect[1]-dot_A_vect[1])
                    # compute ((xC - xA), (yC yA))  
                    pair_2_scalar_vectors = (dot_C_vect[0]-dot_A_vect[0], 
                                            dot_C_vect[1]-dot_A_vect[1])
                    #pairs_dict
                    cosin_sim_dict[cosine_sim_key_entry] = compute_cosine(np.asarray(pair_1_scalar_vectors),
                                                                          np.asarray(pair_2_scalar_vectors))
        return cosin_sim_dict
    
    def print_cosin_sim_dict(self, cosin_sim_dict):
        print("")        
        print("*******************************************************")     
        print("*****  COSINE SIMILARITY ENTRIES **********************")   
        count = 0
        for k,v in cosin_sim_dict.items():
            count+=1
        print(f"number of entries in dict : {count}")
        for k,v in cosin_sim_dict.items():    
            print(f"{k} : {v}")
            
    def print_kpts_to_neighbor_predict(self, kpts_to_neighbor_predict: dict):
        # Display dictionnary of node and its neighbours from skeletons
        for keypoint,keypoint_attr in kpts_to_neighbor_predict.items():
            # display 
            print("----------------------------------------------------")
            print(f"center_node: {keypoint} [{keypoint_attr[keypoint]}]")
            if len(keypoint_attr['neighbours']) < 2:
                continue  
            for k_n, k_attr in keypoint_attr['neighbours'].items():
                print(f"neighbour_node: {k_n} [{k_attr}]")
            #
            number_of_neighbours_ = len(keypoint_attr['neighbours'])
            print(f"number of neighours: {number_of_neighbours_}")   
            
    def run(self):
        cosin_sim_dict = {}
        # kpts_to_neighbor_predict = self.compute_kpts_to_neighbor_from_ref_dictionnary()
        # self.print_kpts_to_neighbor_predict(kpts_to_neighbor_predict)
        self.cosin_sim_dict = self.compute_cosine_similarities()
        #self.print_cosin_sim_dict(cosin_sim_dict)
        return self.cosin_sim_dict
    
    # THE/ZE SCORING ALGORITHM
    def compute_asana_scoring(self, another_scoring_instance: object ) -> dict:
        
        cosin_sim_dict_A = self.cosin_sim_dict
        cosin_sim_dict_B = another_scoring_instance.cosin_sim_dict
        
        # create dictionnary of cosin_sim_diff_ for node['SEGMENT']
        cosin_sim_diff_= {}
        
        for k,v in cosin_sim_dict_A.items():
            if k in cosin_sim_dict_B.keys():
                segment_cos_sim = {}
                node_, pair_ = k.split('-')
                pair_ = pair_.replace('(','').replace(')','')
                pair_1 = pair_.split(', ')[0]
                pair_2 = pair_.split(', ')[1]
                l = sorted([pair_1,pair_2])
                key = f"{node_}-{l[0]}|{l[1]}"
 
                segment_cos_sim['L1'] = compute_cosine_sim_L1( v, cosin_sim_dict_B[k])
                segment_cos_sim['L2'] = compute_cosine_sim_L2( v, cosin_sim_dict_B[k])
                cosin_sim_diff_[key] = segment_cos_sim
                
        # create dictionnary for node['NODE'], a node based information for mean of cosin_sim per nodes
        node_mean_ = {}
        for k,v in cosin_sim_diff_.items():
            # identify carrying node
            current_node = k.split('-')[0]
            # skip if already computed
            if current_node in list(node_mean_.keys()):
                continue
            # count current_node_occurence
            sum_L1 = 0
            sum_L2 = 0
            current_node_occurence = 0
            for k2, v2 in cosin_sim_diff_.items():
                if current_node == k2.split('-')[0]:
                    sum_L1 += v2['L1']
                    sum_L2 += v2['L2']
                    current_node_occurence += 1
            node_mean_[current_node] = {'mean_L1':sum_L1 / current_node_occurence, 
                                        'mean_L2':sum_L2 / current_node_occurence}
            
        # compute global scores for node['GLOBAL']
        mean_L1s = 0
        mean_L2s = 0
        number_of_nodes = len(list(node_mean_.keys()))
        for k, v in node_mean_.items():
            mean_L1s += v['mean_L1']
            mean_L2s += v['mean_L2']
        mean_L1s = mean_L1s / number_of_nodes
        mean_L2s = mean_L2s / number_of_nodes
        
        mean_cosine_similarities_L1 = 0
        mean_cosine_similarities_L2 = 0
        number_of_segments = len(list(node_mean_.keys()))
        for k, v in cosin_sim_diff_.items():
            mean_cosine_similarities_L1 += v['L1']
            mean_cosine_similarities_L2 += v['L2']
        mean_cosine_similarities_L1 = mean_L1s / number_of_segments
        mean_cosine_similarities_L2 = mean_L2s / number_of_segments
        overall_score_ = {}
        overall_score_['mean_L1s'] = mean_L1s
        overall_score_['mean_L2s'] = mean_L2s
        overall_score_['mean_cosine_similarities_L1'] = mean_cosine_similarities_L1
        overall_score_['mean_cosine_similarities_L2'] = mean_cosine_similarities_L2
        
        #collecting dictionnaries into 1
        output_ = {}
        output_['segments'] = cosin_sim_diff_  
        output_['nodes']    = node_mean_
        output_['overall']  = overall_score_
        return output_
        
        
        
        
        

        
        
                    
                    
                

    