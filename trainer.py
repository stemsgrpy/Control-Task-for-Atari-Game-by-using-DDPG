import os
import math
import scipy
import numpy as np
from config import Config
from core.logger import TensorBoardLogger
from core.util import get_output_folder

class Trainer:
    def __init__(self, agent, env, config: Config):
        self.agent = agent
        self.env = env
        self.config = config
        self.SaveImage = False

        if not os.path.exists('./history'):
            os.mkdir('./history')

        # non-Linear epsilon decay
        epsilon_final = self.config.epsilon_min
        epsilon_start = self.config.epsilon
        epsilon_decay = self.config.eps_decay

        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
            -1. * frame_idx / epsilon_decay)

        self.outputdir = get_output_folder(self.config.output, self.config.env)
        self.agent.save_config(self.outputdir)
        self.board_logger = TensorBoardLogger(self.outputdir)

    def train(self, pre_fr=0):
        losses = []
        all_rewards = []
        episode_reward = 0
        ep_num = 0
        is_win = False

        state = self.env.reset() # (1, 84, 84)

        # state = np.reshape([state], (84, 84))
        # state = np.reshape([state.transpose(2,1,0)], (84, 84))

        history = np.stack((state, state, state, state), axis=1) # (1, 4, 84, 84)

        if self.SaveImage:
            scipy.misc.imsave('history/history0.jpg',history[0][0,:,:])
            scipy.misc.imsave('history/history1.jpg',history[0][1,:,:])
            scipy.misc.imsave('history/history2.jpg',history[0][2,:,:])
            scipy.misc.imsave('history/history3.jpg',history[0][3,:,:])
        
        for fr in range(pre_fr + 1, self.config.frames + 1):
            # self.env.render()
            epsilon = self.epsilon_by_frame(fr)

            # action = self.agent.act(state, epsilon)
            action = self.agent.act(history, epsilon)

            next_state, reward, done, _ = self.env.step(action)

            next_history = np.reshape([next_state],(1, 1, 84, 84))
            next_history = np.append(next_history, history[:, :3, :, :], axis=1)
            
            if self.SaveImage:
                scipy.misc.imsave('history/history'+str(fr)+'0.jpg',next_history[0][0,:,:])
                scipy.misc.imsave('history/history'+str(fr)+'1.jpg',next_history[0][1,:,:])
                scipy.misc.imsave('history/history'+str(fr)+'2.jpg',next_history[0][2,:,:])
                scipy.misc.imsave('history/history'+str(fr)+'3.jpg',next_history[0][3,:,:])

            # self.agent.buffer.add(state, action, reward, next_state, done)
            self.agent.buffer.add(history, action, reward, next_history, done)

            state = next_state
            history = next_history
            episode_reward += reward

            loss = 0
            if self.agent.buffer.size() > self.config.batch_size:
                loss = self.agent.learning(fr)
                losses.append(loss)
                self.board_logger.scalar_summary('Loss per frame', fr, loss)

            if fr % self.config.print_interval == 0:
                print("frames: %5d, reward: %5f, loss: %4f episode: %4d, epsilon: %4f" % (fr, np.mean(all_rewards[-10:]), loss, ep_num, self.epsilon_by_frame(fr)))

            if fr % self.config.log_interval == 0:
                self.board_logger.scalar_summary('Reward per episode', ep_num, all_rewards[-1])

            if self.config.checkpoint and fr % self.config.checkpoint_interval == 0:
                self.agent.save_checkpoint(fr, self.outputdir)

            if done:
                state = self.env.reset()

                # state = np.reshape([state], (84, 84))
                # state = np.reshape([state.transpose(2,1,0)], (84, 84))

                history = np.stack((state, state, state, state), axis=1) #(1, 4, 84, 84)
                
                all_rewards.append(episode_reward)
                episode_reward = 0
                ep_num += 1
                avg_reward = float(np.mean(all_rewards[-100:]))
                self.board_logger.scalar_summary('Best 100-episodes average reward', ep_num, avg_reward)

                if len(all_rewards) >= 100 and avg_reward >= self.config.win_reward and all_rewards[-1] > self.config.win_reward:
                    is_win = True
                    self.agent.save_model(self.outputdir, 'best')
                    print('Ran %d episodes best 100-episodes average reward is %3f. Solved after %d trials ✔' % (ep_num, avg_reward, ep_num - 100))
                    if self.config.win_break:
                        break

        if not is_win:
            print('Did not solve after %d episodes' % ep_num)
            self.agent.save_model(self.outputdir, 'last')
