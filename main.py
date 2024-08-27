from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
# from view_transformer import ViewTransformer
# from speed_and_distance_estimator import SpeedAndDistance_Estimator


def main():
    # Read Video
    video_frames = read_video('input_videos/input.mp4')

    # Initialize Tracker
    tracker = Tracker('models/player.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')

    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # # View Trasnformer
    # view_transformer = ViewTransformer()
    # view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # # Speed and distance estimator
    # speed_and_distance_estimator = SpeedAndDistance_Estimator()
    # speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    team_colors = team_assigner.team_colors
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, player in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 player['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_colors[team]
    
    # Assign Goalkeeper Teams
    team_1_centroid, team_2_centroid = team_assigner.get_player_team_centroid(tracks['players'])
    for frame_num, goalkeeper_track in enumerate(tracks['goalkeeper']):
        if goalkeeper_track is not None:
            for goalkeeper_id, goalkeeper in goalkeeper_track.items():
                goalkeeper_team = team_assigner.get_goalkeeper_team(goalkeeper['bbox'],
                                                                    goalkeeper_id,
                                                                    team_1_centroid[frame_num],
                                                                    team_2_centroid[frame_num])
                tracks['goalkeeper'][frame_num][goalkeeper_id]['team'] = goalkeeper_team
                tracks['goalkeeper'][frame_num][goalkeeper_id]['team_color'] = team_colors[goalkeeper_team]


    # Assign Ball Aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            assigned_player = player_assigner.assign_ball_to_player(tracks['goalkeeper'][frame_num], ball_bbox)
            if assigned_player != -1:
                tracks['goalkeeper'][frame_num][assigned_player]['has_ball'] = True
                team_ball_control.append(tracks['goalkeeper'][frame_num][assigned_player]['team'])
            else:
                team_ball_control.append(team_ball_control[-1])

    team_ball_control= np.array(team_ball_control)


    # Draw output 
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control, team_colors)

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    # ## Draw Speed and Distance
    # speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

    # Save video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()