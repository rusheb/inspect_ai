from inspect_ai import Task, eval, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState


@scorer(metrics=[mean()])
def myscorer():
    async def score(state: TaskState, target: Target):
        return Score(value=1)

    return score


@task
def mytask():
    return Task(
        dataset=MemoryDataset([Sample(input="") for _ in range(10)]),
        plan=[],
        scorer=myscorer(),
    )


def test_recompute_scores():
    log = eval(mytask())[0]
    print(log.location)
    pass
