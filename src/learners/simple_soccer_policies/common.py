from src.envs.simple_soccer import SoccerStates


class EvalPolicy:
    def compute_actions(self, states: SoccerStates):
        raise NotImplementedError()
