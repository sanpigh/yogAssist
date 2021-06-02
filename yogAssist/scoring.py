import itertools
import numpy as np

KPTS_TO_NEIGHBOURS_DEF = {  'nose':['sternum'],
                            'sternum':['nose','Rshoulder','Rhip','Lhip','Lshoulder'],
                            'Rshoulder':['Relbow','sternum', 'Lshoulder'],
                            'Lshoulder':['Relbow','sternum', 'Rshoulder'],
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

def compute_cosine_sim(AB, AC):
    # cosine similarity between A and B
    cos_sim=np.dot(AB,AC)/(np.linalg.norm(AB)*np.linalg.norm(AC))
    return cos_sim

class Scoring:
    
    kpts_to_neighbor_dict = KPTS_TO_NEIGHBOURS_DEF
        
    def __init__(self, keypoints):
        self.keypoints = keypoints
    
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
                # if end of list, loop on first element
                idx_next = idx + 1 if idx < number_of_neighbours_ - 1 else 0
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
                cosin_sim_dict[cosine_sim_key_entry] = compute_cosine_sim(np.asarray(pair_1_scalar_vectors),
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
        cosin_sim_dict = self.compute_cosine_similarities()
        #self.print_cosin_sim_dict(cosin_sim_dict)
        return cosin_sim_dict