import torch.optim as optim
from pytorch_dqn.dqn_model import DQN
from dqn_learn import OptimizerSpec
from dqn_class import dqn
from pytorch_dqn.utils.schedule import LinearSchedule
import torch
import environment
# from iterative_astar import iterative_astar
import numpy as np
import argparse
import random
import cv2
import pdb
import json

BATCH_SIZE = 32
#GAMMA = 0.99
#REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS = 50000
#LEARNING_FREQ = 4
#TARGER_UPDATE_FREQ = 10000
#LEARNING_RATE = 0.00025
#ALPHA = 0.95
#EPS = 0.01

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('number_of_games',type=int,help='number of games to play')
    parser.add_argument('--maze_width',type=int,default=50,help='size of maze (default=50)')
    parser.add_argument('--gamma',type=float,default=0.99,help='weighting function for next Q values, default=0.99')
    parser.add_argument('--replay_buffer_size',type=int,default=1000000,help='size of the replay buffer (default=1000000)')
    parser.add_argument('--learning_freq',type=int,default=4,help="how many steps between learning updates (default=4)")
    parser.add_argument('--target_update_freq',type=int,default=10000, help='periodically update the target network by Q network to target Q network (default=10000)')
    parser.add_argument('--learning_rate',type=float,default=0.00025,help='Initial learning rate (default=0.00025)')
    parser.add_argument('--alpha',type=float,default=0.95,help='Alpha value during q-learning (default=0.95)')
    parser.add_argument('--eps',type=float,default=0.01,help='EPS(default=0.1)')
    parser.add_argument('--number_train_games',type=int,default=500,help='Number of games to play before testing')
    parser.add_argument('--number_test_games',type=int,default=20,help='Number of games to play during each testing')
    parser.add_argument('--drawing_frequency',type=int,default=100,help='How frequently to visualize a game')
    parser.add_argument('--prior_model',type=str,default=None,help='Load from a prior pt (default=None)')
    args = parser.parse_args()

    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=args.learning_rate, alpha=args.alpha, eps=args.eps),
    )

    exploration_schedule = LinearSchedule(10000000, 0.01)
    maze_blueprint = np.ones((args.maze_width, args.maze_width),dtype=float)
    #env = environment.dirty_room([20, 40], [20, 50], [10, 20], max_num_steps=1000)
    env = environment.semantic_room([20, 40], [20, 50], clutter_noise=0.3, one_way_stuck_pct=0.1, history_size=10000, max_num_steps=2000)
    # maze = qmaze.Qmaze_hill(maze_blueprint, max_num_steps=500, noise=0.0, base_visibility=5, num_hills=3, max_hill_height=10)

    DQN_model = dqn(
        env=env,
        q_func=DQN,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        replay_buffer_size=args.replay_buffer_size,
        batch_size=BATCH_SIZE,
        gamma=args.gamma,
        learning_starts=LEARNING_STARTS,
        learning_freq=args.learning_freq,
        target_update_freq=args.target_update_freq)
    if args.prior_model is not None:
        DQN_model.Q.load_state_dict(torch.load(args.prior_model))
    torch.save(DQN_model.Q.state_dict(), 'DQN.pt')
    game_count = 0
    results_file="results-tmp.log"
    fin = open(results_file, 'w')
    fin.close()
    while game_count<args.number_of_games:
        DQN_model.train(args.number_train_games,args.drawing_frequency)
        game_count += args.number_train_games
        DQN_performance = 0
        DQN_length = 0
        DQN_success = 0
        #torch.save(DQN_model.Q,'DQN.pt')
        torch.save(DQN_model.Q.state_dict(), 'DQN.pt')
        for te in range(args.number_test_games):
            env.reset(num_solid_objects=np.random.choice([5, 7]))
            DQN_model.play_game(draw_game=((te % 2) == 0), isTrain=False)
            DQN_performance += np.sum(env.episode_rewards)
            DQN_length += len(env.episode_rewards)
            DQN_success += (env.episode_rewards[-1]>0)
            # DQN_path40 += (len(maze.episode_rewards)<40)
            # env.reset(isRedraw=False)
            # iterative_astar(maze, (te % 2) == 0)
            # ASTAR_performance += np.sum(maze.episode_rewards)
            # ASTAR_length += len(maze.episode_rewards)
            # ASTAR_success += (maze.episode_rewards[-1]>0)
            # ASTAR_path40 += (len(maze.episode_rewards)<40)
        print("******** TEST RESULTS (%d) **********" % (game_count))
        # print(" Mean Reward - ASTAR: %f, DQN: %f" % (ASTAR_performance/args.number_test_games, DQN_performance/args.number_test_games))
        print(" Mean Reward - DQN: %f" % (DQN_performance/args.number_test_games))
        print("GC: %d, Path Length: %f, Success Count: %d" % (game_count,DQN_length/args.number_test_games, DQN_success))
        print("******** ***************** **********" )
        with open(results_file, 'a+') as f:
            print("%d, %f, %f, %d" % (game_count, DQN_performance/args.number_test_games, DQN_length / args.number_test_games, DQN_success), file=f)


if __name__ == '__main__':
    main()
