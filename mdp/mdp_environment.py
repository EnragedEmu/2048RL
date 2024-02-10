from abc import *

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class MDPModel:

    @abstractmethod
    def receive(self, **kwargs):
        pass

    @abstractmethod
    def reset(self, **kwargs):
        pass


class Environment(MDPModel):
    def __init__(self, action_set=None,
                 state_init=None, episodic=False, state_terminal=[]):

        self.action_set = action_set
        if self.action_set is None:
            self._set_action()

        self.action_num = len(self.action_set)


        # 에피소딕 MDP의 경우에만 작동합니다.
        self.episodic = episodic
        if self.episodic:
            #for states_t in state_terminal:
            #    assert states_t in 
            self.state_terminal = state_terminal

        # 초기 상태를 지정합니다.
        self.state_init = state_init
        if self.state_init is None:
            self._init_state()
        
        self.state_curr = self.state_init
        self.action_curr = None


    def _set_action(self):
        '''
        self.action_set을 Class의 성격에 맞게 자동으로 지정합니다.
        '''
        raise NotImplementedError()

    def _init_state(self):
        '''
        초기 상태값이 주어지지 않을 경우 랜덤으로 지정합니다.
        어떤 확률로 지정하는지는 상속을 통해 customize할 수 있습니다.
        '''
        raise NotImplementedError()


    def state(self):
        '''
        ***OOP style function***

        Environment의 현재 상태를 return합니다.
        '''
        return self.state_curr


    def dynamics(self, action):
        '''
        Args
            action  -- action instance; currently received action from an agent

        현재 상태(self.state_curr)과 입력된 행동(action)을 이용해
        다음 상태(state_next)와 보상(reward_next)을 계산하는 함수입니다.
        '''
        assert action in self.action_set.items()
        state_next, reward_next = self._dynamics(action)

        return state_next, reward_next

    def dynamics_(self, action):
        '''
        Args
            action  -- action instance; currently received action from an agent

        현재 상태(self.state_curr)과 입력된 행동(action)을 이용해
        최근 행동(self.action_curr)과 현재 상태(self.state_curr)를 바꾸고
        이에 따른 보상(reward)을 return하는 함수입니다.
        '''

        # 입력한 행동이 행동 집합에 포함되는지 확인하고, 최근 행동을 업데이트합니다.
        assert action in self.action_set
        self.action_curr = action

        # self._dynamics()를 이용하여 state와 action에 맞는 다음 단계의 state와 action을 얻습니다.
        state_next, reward_next = self._dynamics(self.state_curr, self.action_curr)

        # 현재 상태를 새로 얻은 상태값으로 변환합니다.
        self.state_curr = state_next

        return reward_next

    def _dynamics(self, action):
        '''
        Args
            state   -- state instance; an arbitrary state for testing
            action  -- action instance; an arbitrary received action from an agent for testing

        상태(state)과 입력된 행동(action)에 따른 다음 상태와 보상을 return하는 함수입니다.
        '''
        raise NotImplementedError()

    def change_dynamics(self, **kwargs):
        '''
        step에 따른 _dynamics 함수의 변형이 필요할 때 implement합니다.
        '''
        raise NotImplementedError()


    def reset(self, state_init=None):
        self.state_curr = self.state_init
        if self.state_init is None:
            if state_init is None:
                self._init_state()
            else:
                self.state_curr = state_init

if __name__ == "__main__":
    try:
        print("main")
    except KeyboardInterrupt:
        print("exiting mdp_environment.py")
        exit(0)