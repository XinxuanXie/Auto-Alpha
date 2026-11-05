import unittest

from evoalpha_workflow import (
    EvoAlphaConfig,
    EvoAlphaTools,
    EvoAlphaWorkflow,
    EvoAlphaWorkflowError,
)
from evoalpha_workflow.state import (
    CandidateProposal,
    EvaluatedCandidate,
    EvoAlphaStage,
    ProposalType,
)


def seed_candidate(index: int) -> EvaluatedCandidate:
    return EvaluatedCandidate(
        factor_id=f"SEED-{index:03d}",
        generation=0,
        proposal_type=ProposalType.SEED,
        composite_score=1.0 - index / 1_000,
    )


class FakeSeedEvaluator:
    def evaluate_seeds(self):
        return [seed_candidate(index) for index in range(40)]


class FakeOffspringProposer:
    def __init__(self, invalid_count: bool = False) -> None:
        self.invalid_count = invalid_count

    def propose_offspring(self, population, generation, config):
        proposals = []
        for index in range(config.refinement_offspring_per_generation):
            proposals.append(
                CandidateProposal(
                    factor_id=f"G{generation}-R{index:02d}",
                    generation=generation,
                    proposal_type=ProposalType.REFINEMENT,
                    parent_factor_ids=(population[index].factor_id,),
                )
            )
        for index in range(config.crossover_tasks_per_generation):
            proposals.append(
                CandidateProposal(
                    factor_id=f"G{generation}-C{index:02d}",
                    generation=generation,
                    proposal_type=ProposalType.CROSSOVER,
                    parent_factor_ids=(
                        population[index].factor_id,
                        population[index + 1].factor_id,
                    ),
                )
            )
            proposals.append(
                CandidateProposal(
                    factor_id=f"G{generation}-M{index:02d}",
                    generation=generation,
                    proposal_type=ProposalType.MUTATION,
                    parent_factor_ids=(f"G{generation}-C{index:02d}",),
                )
            )
        return proposals[:-1] if self.invalid_count else proposals


class FakeCandidateEvaluator:
    def evaluate_candidates(self, proposals, generation):
        return [
            EvaluatedCandidate(
                factor_id=proposal.factor_id,
                generation=generation,
                proposal_type=proposal.proposal_type,
                parent_factor_ids=proposal.parent_factor_ids,
                composite_score=2.0 - index / 1_000,
                artifact=proposal.artifact,
            )
            for index, proposal in enumerate(proposals)
        ]


class FakeValidationEvaluator:
    def evaluate_validation(self, population):
        return [
            candidate.model_copy(
                update={"composite_score": 1.0 - index / 100}
            )
            for index, candidate in enumerate(population)
        ]


class FakeTestEvaluator:
    def evaluate_test(self, selected_candidates):
        return [
            candidate.model_copy(
                update={"composite_score": 0.9 - index / 100}
            )
            for index, candidate in enumerate(selected_candidates)
        ]


def make_tools(invalid_count: bool = False) -> EvoAlphaTools:
    return EvoAlphaTools(
        seed_evaluator=FakeSeedEvaluator(),
        offspring_proposer=FakeOffspringProposer(invalid_count=invalid_count),
        candidate_evaluator=FakeCandidateEvaluator(),
        validation_evaluator=FakeValidationEvaluator(),
        test_evaluator=FakeTestEvaluator(),
    )


class EvoAlphaWorkflowTests(unittest.TestCase):
    def test_runs_fixed_three_generation_protocol(self):
        state = EvoAlphaWorkflow(make_tools()).run(run_id="TEST-RUN")

        self.assertEqual(state.stage, EvoAlphaStage.COMPLETED)
        self.assertEqual(state.generation, 3)
        self.assertEqual(len(state.seed_candidates), 40)
        self.assertEqual(len(state.population), 30)
        self.assertEqual(len(state.current_proposals), 28)
        self.assertEqual(len(state.current_offspring), 28)
        self.assertEqual(len(state.snapshots), 4)
        self.assertEqual([item.generation for item in state.snapshots], [0, 1, 2, 3])
        self.assertEqual(len(state.validation_results), 30)
        self.assertEqual(len(state.test_results), 10)
        self.assertEqual(
            [event.sequence for event in state.events],
            list(range(1, len(state.events) + 1)),
        )
        self.assertEqual(state.events[-1].event_type, "workflow_completed")

    def test_rejects_offspring_count_outside_fixed_protocol(self):
        workflow = EvoAlphaWorkflow(make_tools(invalid_count=True))

        with self.assertRaises(EvoAlphaWorkflowError) as context:
            workflow.run(run_id="FAILED-RUN")

        self.assertEqual(context.exception.state.stage, EvoAlphaStage.FAILED)
        self.assertIn("offspring proposal count", context.exception.state.error)
        self.assertEqual(
            context.exception.state.events[-1].event_type,
            "workflow_failed",
        )

    def test_config_rejects_validation_size_above_population(self):
        with self.assertRaises(ValueError):
            EvoAlphaConfig(population_size=5, validation_selection_size=10)


if __name__ == "__main__":
    unittest.main()
