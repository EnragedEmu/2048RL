from ..mdp_environment import Environment
import numpy as np
import json

class square2048(Environment):

    def __init__(self):
        with open(__file__.replace('py', 'json'), 'r') as f:
            self.config = json.load(f)["environment"]
        print("----initializing environment: square2048------")
        print(self.config)
        print("-" * 50)
        self.square_size = self.config['square_size']
        self.len = self.square_size * self.square_size
        self.initial_block_num = self.config['init_block_num']

        super().__init__(action_set=None, state_init=None,
                         episodic=True, state_terminal=[])

    def _init_state(self):
        state_init_index = np.random.choice(self.len, 
                                            self.initial_block_num, 
                                            replace=False)
        self.state_init = np.zeros(self.len, dtype=np.uint8)
        self.state_init[state_init_index] = 1

    def _set_action(self):
        self.action_set = self.config["action_set"]

    @staticmethod
    def __merge_block_from_x2y(row: np.array, index_x: int, index_y: int) -> int:
        if index_x == index_y:
            print("This should not happen: BUG!!!!!!!!!!!!!")
            breakpoint()
            return 0
        else:
            row[index_y] += 1
            row[index_x] = 0
            return 2 ^ row[index_y]

    @staticmethod
    def __move_block_from_x2y(row: np.array, index_x: int, index_y: int):
        if index_x == index_y:
            return
        else:
            row[index_y] = row[index_x]
            row[index_x] = 0
    
    def __move_row_LEFT_reward(self, row: np.array) -> int:
        curr_new_row_i = 0
        row_reward = 0
        merged_reward = 0

        for i in range(self.square_size):
            if row[i] == 0:
                continue

            check_merge = all(
                [merged_reward == 0,
                 curr_new_row_i > 0,
                 row[curr_new_row_i - 1] == row[i]]
            )

            if check_merge:
                merged_reward = \
                    self.__merge_block_from_x2y(row, i, curr_new_row_i - 1)
            else:
                self.__move_block_from_x2y(i, curr_new_row_i)
                merged_reward = 0
            
            row_reward += merged_reward
        
        return row_reward


    def __move_row_LEFT(self, row: np.array):
        new_row = np.zeros(self.square_size, dtype=np.uint8)
        curr_new_row_i = 0
        just_merged = False

        for i in range(self.square_size):
            if row[i] == 0:
                continue

            check_merge = all(
                [not just_merged, 
                curr_new_row_i > 0, 
                new_row[curr_new_row_i - 1] == row[i]]
            )

            if  check_merge:
                new_row[curr_new_row_i - 1] += 1
                just_merged = True
            else:
                new_row[curr_new_row_i] = row[i]
                just_merged = False
                curr_new_row_i += 1
        
        return new_row
    
    def move_LEFT_and_return_reward(self):
        current_state = \
            self.state_curr.reshape(self.square_size, self.square_size)
        print(current_state)
        reward = 0
        for i, row in enumerate(current_state):
            row_reward = self.__move_row_LEFT_reward(row)
            reward += row_reward


    def move_LEFT_and_return_isChange(self):
        current_state = \
            self.state_curr.reshape(self.square_size, self.square_size)
        print(current_state)
        isChange = False
        for i, row in enumerate(current_state):
            tmp = np.array(row)
            current_state[i] = self.__move_row_LEFT(row)
            if not isChange and (tmp != row).any():
                isChange = True
        print(isChange)

        print(current_state)
        if (not isChange):
            return isChange

        self.__generate_one_block()

        return isChange

    def __generate_one_block(self):
        empty = np.where(self.state_curr == 0)
        if len(empty[0]) == 0:
            self.terminate()
        else:
            generate_index = np.random.choice(empty[0], 1)
            self.state_curr[generate_index] = 1

    def terminate(self):
        print("terminate should be implemented")

    def _dynamics(self, state, action):

        pass


if __name__ == "__main__":
    myclass = square2048()
    for i in range(9):
        myclass.move_LEFT_and_return_isChange()