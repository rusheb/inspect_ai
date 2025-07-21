import pytest

from inspect_ai import Task, eval_async, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.scorer._metric import ScoreEdit, edit_score, recompute_metrics
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


@pytest.mark.asyncio
async def test_recompute_scores():
    logs = await eval_async(mytask())
    log = logs[0]
    print(log.location)

    assert log.results is not None and log.results.scores is not None
    assert log.results.scores[0].metrics["mean"].value == 1

    assert log.samples is not None
    for i, sample in enumerate(log.samples):
        if i % 2 == 0:
            continue

        assert sample.scores is not None
        final_score = sample.scores["myscorer"]
        edit = ScoreEdit(
            value=0,
        )
        await edit_score(log, final_score, edit, should_recompute_metrics=False)

    await recompute_metrics(log)

    for i, sample in enumerate(log.samples):

        assert sample.scores is not None
        final_score = sample.scores["myscorer"]
        if i % 2 == 0:
            assert final_score.value == 1
        else:
            assert final_score.value == 0

    assert log.results.scores[0].metrics["mean"].value == 0.5
