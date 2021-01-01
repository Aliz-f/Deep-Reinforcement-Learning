from learn.RL import *
from base.base import BaseAgent, Action, TurnData
from utils import *


class LearnerAgent(BaseAgent):
    initial = True

    def do_turn(self, turn_data: TurnData) -> Action:
        state = turn_data
        if self.initial:
            np_map, diamonds, bases = generate_map(state.map)
            env = TdfMaze(np_map, diamonds, bases, self.max_turns)
            agent_position = [state.agent_data[0].position]
            model = build_model(env)
            self.qt = Qtraining(
                model,
                env,
                n_epoch=100,
                max_memory=500,
                data_size=100,
                name="model_1",
                agent_cells=agent_position
            )
            self.qt.train()

            self.qt.save('model1')
            print('Compute Time: ', self.qt.seconds)
            self.initial = False
        return self.qt.play_game()


if __name__ == '__main__':
    winner = LearnerAgent().play()
    print("Winner: ", winner)
