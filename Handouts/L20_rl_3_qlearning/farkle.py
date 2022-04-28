import abc
import copy
from collections import Counter
import random
import time
from typing import Dict, List, Tuple, NamedTuple, Optional


class Action(NamedTuple):
    used: Tuple
    name: str
    value: int

    def __str__(self):
        if self.name.lower() in ["roll", "stop"]:
            return self.name
        else:
            return f"Play {self.name} to score {self.value}"


BANKRUPT = Action(tuple(), "bankrupt", 0)
ROLL = Action(tuple(), "roll", 0)
STOP = Action(tuple(), "stop", 0)


class Dice(object):
    """
    A 6-sided dice object that can be used in dice games implemented
    in Python

    The dice can take the values 1, 2, 3, 4, 5, 6

    The dice keeps track of the most recently rolled value in the
    `value` property. This value is `None` until the dice is rolled.

    Methods
    -------
    roll :
        Rolls the dice and gives a random integer
    """

    def __init__(self, value: Optional[int] = None):
        self.value = value
        self.unicode_dice = {
            1: "\u2680",
            2: "\u2681",
            3: "\u2682",
            4: "\u2683",
            5: "\u2684",
            6: "\u2685",
        }
        if value is None:
            self.roll()

    def __eq__(self, other: "Dice"):
        return self.value == other.value

    def __lt__(self, other: "Dice"):
        return self.value < other.value

    def __repr__(self):
        if self.value is None:
            msg = "The dice has not been rolled"
        else:
            msg = f"{self.unicode_dice[self.value]}"

        return msg

    def roll(self):
        """
        Rolls the dice and reset the current value

        Returns
        -------
        value : int
            The number rolled by the dice
        """
        value = random.randint(1, 6)
        self.value = value

        return value


class State:
    # public game state
    current_round: int
    scores: List[int]
    can_roll: int
    rolled_dice: List[Dice]
    turn_sum: int

    # internal state
    _n_players: int

    def __init__(self, n_players):
        self._n_players = n_players
        self.current_round = 0
        self.scores = [0] * self._n_players
        self.can_roll = 6
        self.rolled_dice = []
        self.turn_sum = 0

    def __dir__(self):
        return [
            "current_round",
            "scores",
            "can_roll",
            "rolled_dice",
            "turn_sum",
        ]

    @property
    def __dict__(self) -> dict:
        return {k: getattr(self, k) for k in dir(self)}

    @__dict__.setter
    def __dict__(self, val: dict):
        for (k, v) in val.items():
            setattr(self, k, copy.deepcopy(v))

    def __repr__(self):
        return f"Round: {self.current_round}. Score: {self.scores}"

    def __copy__(self) -> "State":
        out = State(self._n_players)
        out.__dict__ = self.__dict__
        return out

    def observable_state(self):
        return (
            self.can_roll,
            self.turn_sum,
            tuple(d.value for d in self.rolled_dice),
        )

    @property
    def current_player(self) -> int:
        return self.current_round % self._n_players

    def end_turn(self, forced: bool = False) -> "State":
        """
        End the turn for the current player

        If the player was not forced to end, the `turn_sum` is added to their score

        Parameters
        ----------
        forced: bool, default=False
            A boolean indicating if the player was forced to end his/her turn. If true,
            the player does not get any points for this turn.

        Returns
        -------
        new_state: State
            A new instance of the state is returned recording updated score,
            refreshed dice, and reset current sum
        """

        out = State(self._n_players)
        out.current_round = self.current_round + 1
        out.scores = copy.deepcopy(self.scores)

        if not forced:
            out.scores[self.current_player] += self.turn_sum
        return out

    def roll(self) -> "State":
        out = copy.copy(self)
        out.rolled_dice = sorted([Dice() for _ in range(self.can_roll)])
        out.can_roll = 0  # mark that only actions are to consider scores
        return out

    def play_dice(self, action: Action) -> "State":
        # TODO: validate that the chosen dice exist
        out = copy.copy(self)

        # add value to the current sum
        out.turn_sum += action.value

        n_played = len(action.used)

        # update number of dice that can be rolled
        out.can_roll = len(self.rolled_dice) - n_played
        if out.can_roll == 0:
            # can pick them all up!
            out.can_roll = 6

        # Remove the played dice from `rolled_dice`
        for k in action.used:
            out.rolled_dice.remove(Dice(k))

        return out

    def enumerate_options(
        self, rolled_dice: Optional[List[Dice]] = None
    ) -> List[Action]:
        """
        Given a list of dice, it computes all of the possible ways
        that one can score

        Parameters
        ----------
        rolled_dice: Optional[List[Dice]]
            A list of dice for which to enumerate options. If None are passed
            then `self.rolled_dice` is used

        Returns
        -------
        opportunities : List[Action]
            A list of valid actions for a player
        """
        rolled: List[Dice] = self.rolled_dice
        if rolled_dice is not None:
            rolled = rolled_dice

        dice_counts = Counter([d.value for d in rolled])
        opportunities: List[Action] = []

        # Single dice opportunities
        if dice_counts[1] > 0:
            opportunities.append(Action((1,), "1", 100))

        if dice_counts[5] > 0:
            opportunities.append(Action((5,), "5", 50))

        # Three pairs
        pairs = []
        pairs_playable = []
        for i in range(1, 7):
            if dice_counts[i] >= 2:
                pairs.append(i)
                pairs_playable.extend([i] * 2)
        if len(pairs) == 3:
            opportunities.append(Action(tuple(pairs_playable), f"Three pairs", 1500))

        # Three of a kind
        if dice_counts[1] >= 3:
            opportunities.append(Action((1, 1, 1), "Three 1's", 1000))
        for i in range(2, 7):
            if dice_counts[i] >= 3:
                opportunities.append(Action((i, i, i), f"Three {i}'s", i * 100))

        for i in range(1, 7):
            # Four of a kind
            if dice_counts[i] >= 4:
                opportunities.append(Action(tuple([i] * 4), f"Four {i}'s", 1000))

            # Five of a kind
            if dice_counts[i] >= 5:
                opportunities.append(Action(tuple([i] * 5), f"Five {i}'s", 2000))

            # Six of a kind
            if dice_counts[i] == 6:
                opportunities.append(Action(tuple([i] * 6), f"Six {i}'s", 3000))

        # Straight
        if all([dice_counts[i] > 0 for i in range(1, 7)]):
            opportunities.append(Action(tuple(range(1, 7)), "1-2-3-4-5-6", 3000))

        # can_roll is zero iff I just rolled. If there are no opportunities, we
        # must be bankrupt for this round
        if len(opportunities) == 0 and self.can_roll == 0:
            # oops
            return [BANKRUPT]
        
        if self.turn_sum > 0 and len(opportunities) == 0:
            opportunities.append(STOP)

        if self.can_roll > 0:
            opportunities.append(ROLL)

        return opportunities

    def step(self, action: Action) -> "State":
        if action is STOP:
            new_state = self.end_turn(forced=False)
        elif action is ROLL:
            new_state = self.roll()
        elif action is BANKRUPT:
            new_state = self.end_turn(forced=True)
        else:
            # otherwise the player used some dice
            new_state = self.play_dice(action)

        return new_state


class FarklePlayer(abc.ABC):
    name: str

    @abc.abstractmethod
    def act(self, state: State, choices: List[Action]) -> Action:
        pass

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class RandomFarklePlayer(FarklePlayer):
    name = "random_robot"

    def act(self, state: State, choices: List[Action]) -> Action:
        return random.choice(choices)


class HumanFarklePlayer(FarklePlayer):
    def __init__(self, name: str):
        self.name = name

    def act(self, state: State, choices: List[Action]) -> Action:
        # get an action from the user
        print(
            f"(score: {state.scores}) You have rolled: ",
            state.rolled_dice,
            "(",
            [x.value for x in state.rolled_dice],
            ")",
        )
        print("The current total you have at stake is:", state.turn_sum)
        print("Your scoring options are as follows:")
        for num, act in enumerate(choices):
            print(f"\t{num}: {str(act)}")

        while True:
            msg = "Choose an integer to select a scoring option: "
            try:
                choice = int(input(msg))
                return choices[choice]
            except ValueError:
                print("Input not understood, try again!")


class Farkle(object):
    """
    The Farkle game is executed from this class
    """

    def __init__(self, players, points_to_win=10_000, verbose: bool = False):
        self.points_to_win = points_to_win
        self.players = players
        self._have_human = any(map(lambda x: isinstance(x, HumanFarklePlayer), players))
        self.verbose = verbose or self._have_human
        self.n_players = len(players)
        self._state = State(self.n_players)
        self._history: List[Tuple[State, Action]] = []

    @property
    def state(self) -> State:
        return self._state

    def set_state(self, action: Action, new_state: State):
        self._history.append((self.state, action))
        self._state = new_state

    def reset(self):
        self._state = State(self.n_players)
        self._history = []

    def step(self, action: Action) -> State:
        new_state = self.state.step(action)
        self.set_state(action, new_state)
        return new_state

    def player_turn(self, choices: Optional[List[Action]] = None):
        """
        Lets each player play a turn and then prints the updated
        scores at the end of the turn
        """
        current_player_num = self.state.current_player
        current_player = self.players[current_player_num]

        if choices is None:
            choices = self.state.enumerate_options()
        if len(choices) == 0:
            # bankrupt... bummer
            self.step(BANKRUPT)
            return
        action = current_player.act(self.state, choices)
        self.step(action)

        # check if player chose to stop
        if current_player_num != self.state.current_player:
            return

        # check if player has no actions
        next_choices = self.state.enumerate_options()
        if len(next_choices) == 0:
            self.step(BANKRUPT)
            return

        # otherwise, let the player continue the turn
        self.player_turn(next_choices)

        return None

    def _print_score(self):
        for i in range(self.n_players):
            print(f"  - {self.players[i].name}: {self.state.scores[i]}")

    def play(self):
        """Play a game of Farkle"""
        while True:
            # check end_game
            winners = {
                i: score >= self.points_to_win
                for i, score in enumerate(self.state.scores)
            }

            if any(winners.values()):
                if self.verbose:
                    print("Game over! Final score is:")
                    self._print_score()
                return winners

            # otherwise the game is on!
            if self.verbose:
                print(f"Starting of turn {self.state.current_round}")
                print("Current score:")
                self._print_score()
                time.sleep(0.1)

            for _ in range(self.n_players):
                if self.verbose:
                    current_player = self.players[self.state.current_player]
                    print(f"It is {current_player}'s turn")
                self.step(Action({}, "roll", 0))
                self.player_turn()


class FarkleEnv:
    # first, some helper methods
    def __init__(
        self,
        opponent: FarklePlayer = RandomFarklePlayer(),
        points_to_win=10_000,
        track_history: bool = False,
    ):
        self.points_to_win = points_to_win
        self.opponent = opponent
        self.n_players = 2
        self._state = State(self.n_players)
        self._history: List[Tuple[State,Action]] = []
        self.track_history = track_history
    
    @property
    def state(self) -> State:
        return self._state

    def set_state(self, action: Action, new_state: State):
        self._state = new_state
        if self.track_history:
            self._history.append((self.state, action))

    def opponent_turn(self, s: State) -> State:
        choices = s.enumerate_options()
        action = self.opponent.act(s, choices)
        sp = s.step(action)

        # check if player chose to stop
        if sp.current_player != 1:
            return sp

        # Player didn't stop, but still their turn. Call again
        return self.opponent_turn(sp)

    # key methods needed
    def done(self, state) -> bool:
        return any(score > self.points_to_win for score in state.scores)

    def reset(self):
        self._history = []
        self._state = State(self.n_players)
        return self.state.roll()

    def step(self, s: State, a: Action) -> Tuple[State, int]:
        sp = s.step(a)
        r = 0

        # see if we ended
        if sp.current_player != 0:
            if a is STOP:
                # only score when we choose to stop
                r = s.turn_sum

            # take opponent turn
            sp = self.opponent_turn(sp)

        self.set_state(a, sp)
        return sp, r

    def enumerate_options(self, s: State) -> List[Action]:
        return self.state.enumerate_options(s.rolled_dice)


def play_game(algo):
    algo.restart_episode()
    while not algo.done():
        algo.step()
    return algo


def play_many_games(algo, N):
    terminal_states = []
    print_skip = N // 10
    for i in range(N):
        play_game(algo)
        terminal_states.append(algo.s)
        if i % print_skip == 0:
            print(f"Done with {i}/{N} (len(Q) = {len(algo.Q.Q)})")
    return terminal_states


if __name__ == "__main__":
    # p1 = HumanFarklePlayer("Spencer")
    # p2 = HumanFarklePlayer("Chase")
    p1 = RandomFarklePlayer()
    p2 = RandomFarklePlayer()
    f = Farkle([p1, p2])
    f.play()
