snn
python simulate_goal_visual_encoder_bsn.py --input data/collected_128.npz --out output/encoded_x_128frame_bsn.pt --batch_size 64 --device cuda:0
ann
python simulate_goal_visual_encoder_ann.py --input data/collected_128.npz --out output/encoded_x_128frame_ann.pt --batch_size 64 --device cuda:0