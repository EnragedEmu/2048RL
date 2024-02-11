from ..mdp_environment import Environment
import numpy as np
import json

class square2048(Environment):

    def __init__(self):
        with open(__file__.replace('py', 'json'), 'r') as f:
            self.config = json.load(f)["environment"]
        print("----initializing environment: square2048----")
        print(self.config)
        print('\n')
        self.square_size = self.config['square_size']
        self.len = self.square_size * self.square_size
        self.initial_block_num = self.config['init_block_num']
        
        if self.initial_block_num > self.len:
            print("init_block_num is too big.\n Resizing it to 2\n")
            self.initial_block_num = 2

        self.np_dtype = getattr(np, self.config["np_dtype"])

        super().__init__(action_set=None, state_init=None,
                         episodic=True, state_terminal=[])

    def _init_state(self):
        state_init_index = np.random.choice(self.len, 
                                            self.initial_block_num, 
                                            replace=False)
        self.state_init = np.zeros(self.len, dtype=self.np_dtype)
        self.state_init[state_init_index] = 1

    def _set_action(self):
        self.action_set = self.config["action_set"]

    def dynamics_predict(self, action, state = None):
        '''
        Args
            action  -- action instance; currently received action from an agent

        현재 상태(self.state_curr)과 입력된 행동(action)을 이용해
        다음 상태(state_next)와 보상(reward_next)을 계산하는 함수입니다.
        '''
        assert action in self.action_set.values()
        if state is None:
            self.state_dynamics = np.array(self.state_curr, dtype=self.np_dtype)
        else:
            self.state_dynamics = np.array(state, dtype=self.np_dtype)
        reward_next = self._dynamics(action)

        return self.state_dynamics, reward_next

    
    def _dynamics(self, action):
        # choose a function depending on the input action
        func_string = "_move_row_" + list(self.action_set.keys())[action]
        func = getattr(self, func_string)

        # Move
        current_state = \
            self.state_dynamics.reshape(self.square_size, self.square_size)
        reward = 0
        isChange = False
        for i, row in enumerate(current_state):
            row_reward, row_isChange = func(row)
            reward += row_reward
            isChange = isChange or row_isChange
        
        # Generate random block
        if isChange:
            self.__random_generate_one_block()
            return reward
        else:
            assert reward == 0
            return -1
    

    @staticmethod
    def __merge_block_from_x2y(row: np.array, index_x: int, 
                               index_y: int) -> tuple[int, bool]:
        if index_x == index_y:
            print("This should not happen: BUG!!!!!!!!!!!!!")
            breakpoint()
            return 0, False
        else:
            row[index_y] += 1
            row[index_x] = 0
            return 1 << row[index_y], True

    @staticmethod
    def __move_block_from_x2y(row: np.array, index_x: int, 
                              index_y: int) -> tuple[int, bool]:
        if index_x == index_y:
            return 0, False
        else:
            row[index_y] = row[index_x]
            row[index_x] = 0
            return 0, True
    
    def _move_row_LEFT(self, row: np.array) -> tuple[int, bool]:
        """ A function that moves all the blocks in a row left 
        and return reward and whether there is a change

        Args:
            row: 1d numpy array.
        
        Returns:
            row_reward: the sum of all rewards in a row after merged.
            isChange: a boolean type whether there is any change to 
                all blocks in a row. This is used to determine generating
                random blocks after movement.
        """
        curr_new_row_i = 0
        row_reward = 0
        merged_reward = 0
        isChange = False

        for i in range(self.square_size):
            if row[i] == 0:
                continue

            check_merge = (merged_reward == 0) \
                and (curr_new_row_i > 0) \
                and (row[curr_new_row_i - 1] == row[i])

            if check_merge:
                merged_reward, column_change = \
                    self.__merge_block_from_x2y(row, i, curr_new_row_i - 1)
            else:
                merged_reward, column_change = \
                    self.__move_block_from_x2y(row, i, curr_new_row_i)
                curr_new_row_i += 1
            
            row_reward += merged_reward
            isChange = isChange or column_change
        
        return row_reward, isChange

    def __move_row_RIGHT(self, row: np.array) -> tuple[int, bool]:
        """ A function that moves all the blocks in a row right 
        and return reward and whether there is a change

        Args:
            row: 1d numpy array.
        
        Returns:
            row_reward: the sum of all rewards in a row after merged.
            isChange: a boolean type whether there is any change to 
                all blocks in a row. This is used to determine generating
                random blocks after movement.
        """
        curr_new_row_i = self.square_size - 1
        row_reward = 0
        merged_reward = 0
        isChange = False

        for i in range(self.square_size - 1, -1, -1):
            if row[i] == 0:
                continue

            check_merge = (merged_reward == 0) \
                and (curr_new_row_i + 1 < self.square_size) \
                and (row[curr_new_row_i + 1] == row[i])
            if check_merge:
                merged_reward, column_change = \
                    self.__merge_block_from_x2y(row, i, curr_new_row_i + 1)
            else:
                merged_reward, column_change = \
                    self.__move_block_from_x2y(row, i, curr_new_row_i)
                curr_new_row_i -= 1
            
            row_reward += merged_reward
            isChange = isChange or column_change
        
        return row_reward, isChange

    def __random_generate_one_block(self):
        empty = np.where(self.state_dynamics == 0)
        if len(empty[0]) == 0:
            self.terminate()
        else:
            generate_index = np.random.choice(empty[0], 1)
            self.state_dynamics[generate_index] = 1

    def terminate(self):
        print("terminate should be implemented")
    
    def visualize_state(self):
        state = self.state_curr.reshape(self.square_size,
                                        self.square_size)
        print(" " + "- " * self.square_size + " ")
        for row in state:
            print("|", end='')
            for col in row:
                real = ' ' if col == 0 else 1 << col
                print(real, end= ' ')
            print("|")
        print(" " + "- " * self.square_size + " ")


if __name__ == "__main__":
    myclass = square2048()
    for i in range(9):
        myclass.visualize_state()
        print("reward: " + str(myclass.dynamics_(2)) + "\n")