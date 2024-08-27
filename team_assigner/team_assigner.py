from sklearn.cluster import KMeans
import numpy as np
import sys 
sys.path.append('../')
from utils import get_foot_position

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.goalkeeper_team_dict = {}
    
    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        top_half_image = image[0:int(image.shape[0]/2),:]

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels forr each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color


    def assign_team_color(self,frame, player_detections):
        
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color =  self.get_player_color(frame,bbox)
            player_colors.append(player_color)
        
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]


    def get_player_team(self,frame,player_bbox,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame,player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id+=1

        self.player_team_dict[player_id] = team_id

        return team_id

    def get_player_team_centroid(self,players):
        team_1_centroid = {}
        team_2_centroid = {}
        for frame_num, player_track in enumerate(players):
            team_1_xy = {'x': [], 'y': []}
            team_2_xy = {'x': [], 'y': []}
            for player_id, player in player_track.items():
                if players[frame_num][player_id]['team'] == 1:
                    team_1_xy['x'].append(players[frame_num][player_id]['position'][0])
                    team_1_xy['y'].append(players[frame_num][player_id]['position'][1])
                else:
                    team_2_xy['x'].append(players[frame_num][player_id]['position'][0])
                    team_2_xy['y'].append(players[frame_num][player_id]['position'][1])
            
            team_1_centroid[frame_num] = np.array([sum(team_1_xy['x']) / len(team_1_xy['x']), sum(team_1_xy['y']) / len(team_1_xy['y'])])
            team_2_centroid[frame_num] = np.array([sum(team_2_xy['x']) / len(team_2_xy['x']), sum(team_2_xy['y']) / len(team_2_xy['y'])])
        
        return team_1_centroid, team_2_centroid
    
    def get_goalkeeper_team(self,goalkeeper_bbox,goalkeeper_id,team_1_centroid,team_2_centroid):
        if goalkeeper_id in self.goalkeeper_team_dict:
            return self.goalkeeper_team_dict[goalkeeper_id]

        goalkeeper_x, goalkeeper_y = get_foot_position(goalkeeper_bbox)
        goalkeeper_centroid = np.array([goalkeeper_x, goalkeeper_y])

        dist_0 = np.linalg.norm(goalkeeper_centroid - team_1_centroid)
        dist_1 = np.linalg.norm(goalkeeper_centroid - team_2_centroid)

        team_id = 1 if dist_0 < dist_1 else 2

        self.goalkeeper_team_dict[goalkeeper_id] = team_id

        return team_id