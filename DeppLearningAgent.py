from learn.RL import *
from base.base import BaseAgent, Action, TurnData
from utils import *


class LearnerAgent(BaseAgent):

    def do_turn(self, turn_data: TurnData) -> Action:
        state = turn_data
        np_map, diamonds, bases = generate_map(state.map)
        env = TdfMaze(maps, diamonds, bases)
        agent_position = [state.agent_data[0].position]
        model = build_model(env)
        qt = Qtraining(
            model,
            env,
            n_epoch=10,
            max_memory=500,
            data_size=100,
            name="model_1",
            agent_cells=agent_position
        )
        qt.train()

        qt.save('model1')
        print('Compute Time: ', qt.seconds)

        qt.run_game(agent_position)
