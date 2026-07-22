import pytest

from mini_trainer.train import ValidationScheduler


class TestValidationSchedulerIsConfigured:
    def test_no_triggers(self):
        vs = ValidationScheduler()
        assert not vs.is_configured

    def test_step_trigger(self):
        vs = ValidationScheduler(validation_frequency=10)
        assert vs.is_configured

    def test_frequency_zero_not_configured(self):
        vs = ValidationScheduler(validation_frequency=0)
        assert not vs.is_configured

    def test_epoch_trigger(self):
        vs = ValidationScheduler(validate_at_epoch=True)
        assert vs.is_configured

    def test_samples_trigger(self):
        vs = ValidationScheduler(min_samples_per_validation=100)
        assert vs.is_configured

    def test_final_trigger(self):
        vs = ValidationScheduler(validate_at_final=True)
        assert vs.is_configured


class TestStepValidation:
    def test_validates_at_frequency(self):
        vs = ValidationScheduler(validation_frequency=5)
        assert vs.should_validate("step", step=5)
        assert vs.should_validate("step", step=10)
        assert vs.should_validate("step", step=15)

    def test_does_not_validate_between_steps(self):
        vs = ValidationScheduler(validation_frequency=5)
        assert not vs.should_validate("step", step=1)
        assert not vs.should_validate("step", step=3)
        assert not vs.should_validate("step", step=7)

    def test_no_frequency_means_no_step_validation(self):
        vs = ValidationScheduler()
        assert not vs.should_validate("step", step=5)

    def test_frequency_zero_means_no_validation(self):
        vs = ValidationScheduler(validation_frequency=0)
        assert not vs.should_validate("step", step=0)

    def test_step_zero_does_not_validate(self):
        vs = ValidationScheduler(validation_frequency=5)
        assert not vs.should_validate("step", step=0)


class TestEpochValidation:
    def test_validates_at_epoch_boundary(self):
        vs = ValidationScheduler(validate_at_epoch=True)
        assert vs.should_validate("epoch", accumulated_samples=100, end_of_epoch=True)

    def test_does_not_validate_mid_epoch(self):
        vs = ValidationScheduler(validate_at_epoch=True)
        assert not vs.should_validate("epoch", accumulated_samples=100, end_of_epoch=False)

    def test_disabled_means_no_epoch_validation(self):
        vs = ValidationScheduler(validate_at_epoch=False)
        assert not vs.should_validate("epoch", accumulated_samples=100, end_of_epoch=True)

    def test_does_not_double_validate_at_same_samples(self):
        vs = ValidationScheduler(validate_at_epoch=True)
        assert vs.should_validate("epoch", accumulated_samples=100, end_of_epoch=True)
        vs.record_validation("epoch", 100)
        assert not vs.should_validate("epoch", accumulated_samples=100, end_of_epoch=True)

    def test_validates_again_after_new_samples(self):
        vs = ValidationScheduler(validate_at_epoch=True)
        vs.record_validation("epoch", 100)
        assert vs.should_validate("epoch", accumulated_samples=200, end_of_epoch=True)


class TestSamplesValidation:
    def test_validates_after_enough_samples(self):
        vs = ValidationScheduler(min_samples_per_validation=100)
        assert vs.should_validate("samples", accumulated_samples=100)

    def test_does_not_validate_before_threshold(self):
        vs = ValidationScheduler(min_samples_per_validation=100)
        assert not vs.should_validate("samples", accumulated_samples=50)

    def test_validates_again_after_recording(self):
        vs = ValidationScheduler(min_samples_per_validation=100)
        vs.record_validation("samples", 100)
        assert not vs.should_validate("samples", accumulated_samples=150)
        assert vs.should_validate("samples", accumulated_samples=200)

    def test_no_min_samples_means_no_validation(self):
        vs = ValidationScheduler()
        assert not vs.should_validate("samples", accumulated_samples=1000)

    def test_records_sample_based_tracker(self):
        vs = ValidationScheduler(min_samples_per_validation=100)
        vs.record_validation("samples", 100)
        assert vs.last_sample_based_validation_samples == 100
        assert vs.last_validated_samples == 100


class TestFinalValidation:
    def test_validates_at_end_of_training(self):
        vs = ValidationScheduler(validate_at_final=True)
        assert vs.should_validate("final", accumulated_samples=100, end_of_training=True)

    def test_does_not_validate_before_end(self):
        vs = ValidationScheduler(validate_at_final=True)
        assert not vs.should_validate("final", accumulated_samples=100, end_of_training=False)

    def test_disabled_means_no_final_validation(self):
        vs = ValidationScheduler(validate_at_final=False)
        assert not vs.should_validate("final", accumulated_samples=100, end_of_training=True)

    def test_does_not_double_validate(self):
        vs = ValidationScheduler(validate_at_final=True)
        vs.record_validation("final", 100)
        assert not vs.should_validate("final", accumulated_samples=100, end_of_training=True)


class TestRecordValidation:
    def test_record_updates_last_validated_samples(self):
        vs = ValidationScheduler(validation_frequency=5)
        vs.record_validation("step", 50)
        assert vs.last_validated_samples == 50

    def test_record_samples_updates_both_trackers(self):
        vs = ValidationScheduler(min_samples_per_validation=100)
        vs.record_validation("samples", 200)
        assert vs.last_validated_samples == 200
        assert vs.last_sample_based_validation_samples == 200

    def test_record_non_samples_does_not_update_sample_tracker(self):
        vs = ValidationScheduler(validation_frequency=5)
        vs.record_validation("step", 50)
        assert vs.last_sample_based_validation_samples == 0

    def test_record_epoch_updates_epoch_tracker(self):
        vs = ValidationScheduler(validate_at_epoch=True)
        vs.record_validation("epoch", 100)
        assert vs.last_epoch_validated_samples == 100
        assert vs.last_final_validated_samples == 0

    def test_record_final_updates_final_tracker(self):
        vs = ValidationScheduler(validate_at_final=True)
        vs.record_validation("final", 200)
        assert vs.last_final_validated_samples == 200
        assert vs.last_epoch_validated_samples == 0


class TestUnknownValidationType:
    def test_raises_on_unknown_type(self):
        vs = ValidationScheduler()
        with pytest.raises(ValueError, match="Unknown validation type"):
            vs.should_validate("unknown", step=1)


class TestMultipleTriggers:
    def test_step_and_epoch_both_active(self):
        vs = ValidationScheduler(validation_frequency=5, validate_at_epoch=True)
        assert vs.is_configured
        assert vs.should_validate("step", step=5)
        assert vs.should_validate("epoch", accumulated_samples=100, end_of_epoch=True)

    def test_triggers_are_independent(self):
        vs = ValidationScheduler(validation_frequency=5, min_samples_per_validation=100)
        vs.record_validation("step", 50)
        assert vs.should_validate("samples", accumulated_samples=100)

    def test_step_validation_does_not_suppress_epoch(self):
        vs = ValidationScheduler(validation_frequency=5, validate_at_epoch=True)
        vs.record_validation("step", 100)
        assert vs.should_validate("epoch", accumulated_samples=100, end_of_epoch=True)

    def test_step_validation_does_not_suppress_final(self):
        vs = ValidationScheduler(validation_frequency=5, validate_at_final=True)
        vs.record_validation("step", 100)
        assert vs.should_validate("final", accumulated_samples=100, end_of_training=True)

    def test_epoch_record_does_not_suppress_final(self):
        vs = ValidationScheduler(validate_at_epoch=True, validate_at_final=True)
        vs.record_validation("epoch", 100)
        assert vs.should_validate("final", accumulated_samples=100, end_of_training=True)


class TestBoundsValidation:
    def test_min_samples_zero_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            ValidationScheduler(min_samples_per_validation=0)

    def test_min_samples_negative_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            ValidationScheduler(min_samples_per_validation=-1)

    def test_frequency_negative_normalizes_to_none(self):
        vs = ValidationScheduler(validation_frequency=-1)
        assert vs.validation_frequency is None
        assert not vs.is_configured


class TestStateSerializationAndRestore:
    def test_state_roundtrip(self):
        vs = ValidationScheduler(
            validation_frequency=5,
            validate_at_epoch=True,
            min_samples_per_validation=100,
            validate_at_final=True,
        )
        vs.record_validation("step", 50)
        vs.record_validation("samples", 200)
        vs.record_validation("epoch", 300)
        vs.record_validation("final", 400)

        state = {
            "last_validated_samples": vs.last_validated_samples,
            "last_sample_based_validation_samples": vs.last_sample_based_validation_samples,
            "last_epoch_validated_samples": vs.last_epoch_validated_samples,
            "last_final_validated_samples": vs.last_final_validated_samples,
        }

        vs2 = ValidationScheduler(
            validation_frequency=5,
            validate_at_epoch=True,
            min_samples_per_validation=100,
            validate_at_final=True,
        )
        vs2.last_validated_samples = state["last_validated_samples"]
        vs2.last_sample_based_validation_samples = state["last_sample_based_validation_samples"]
        vs2.last_epoch_validated_samples = state["last_epoch_validated_samples"]
        vs2.last_final_validated_samples = state["last_final_validated_samples"]

        assert vs2.last_validated_samples == 400
        assert vs2.last_sample_based_validation_samples == 200
        assert vs2.last_epoch_validated_samples == 300
        assert vs2.last_final_validated_samples == 400

        assert not vs2.should_validate("samples", accumulated_samples=250)
        assert vs2.should_validate("samples", accumulated_samples=300)
        assert not vs2.should_validate("epoch", accumulated_samples=300, end_of_epoch=True)
        assert vs2.should_validate("epoch", accumulated_samples=301, end_of_epoch=True)

    def test_restore_with_defaults_for_missing_keys(self):
        state = {
            "last_validated_samples": 100,
            "last_sample_based_validation_samples": 100,
        }

        vs = ValidationScheduler(validate_at_epoch=True, validate_at_final=True)
        vs.last_validated_samples = state["last_validated_samples"]
        vs.last_sample_based_validation_samples = state["last_sample_based_validation_samples"]
        vs.last_epoch_validated_samples = state.get("last_epoch_validated_samples", 0)
        vs.last_final_validated_samples = state.get("last_final_validated_samples", 0)

        assert vs.last_epoch_validated_samples == 0
        assert vs.last_final_validated_samples == 0
        assert vs.should_validate("epoch", accumulated_samples=1, end_of_epoch=True)
        assert vs.should_validate("final", accumulated_samples=1, end_of_training=True)


class TestCoalescedValidation:
    def test_step_and_sample_coalesce(self):
        vs = ValidationScheduler(validation_frequency=5, min_samples_per_validation=100)
        step_fires = vs.should_validate("step", step=5, accumulated_samples=100)
        sample_fires = vs.should_validate("samples", accumulated_samples=100)
        assert step_fires
        assert sample_fires
        vs.record_validation("step", 100)
        vs.record_validation("samples", 100)
        assert vs.last_validated_samples == 100
        assert vs.last_sample_based_validation_samples == 100

    def test_step_fires_sample_does_not(self):
        vs = ValidationScheduler(validation_frequency=5, min_samples_per_validation=200)
        assert vs.should_validate("step", step=5, accumulated_samples=50)
        assert not vs.should_validate("samples", accumulated_samples=50)

    def test_sample_fires_step_does_not(self):
        vs = ValidationScheduler(validation_frequency=5, min_samples_per_validation=100)
        assert not vs.should_validate("step", step=3, accumulated_samples=100)
        assert vs.should_validate("samples", accumulated_samples=100)

    def test_neither_fires(self):
        vs = ValidationScheduler(validation_frequency=5, min_samples_per_validation=200)
        assert not vs.should_validate("step", step=3, accumulated_samples=50)
        assert not vs.should_validate("samples", accumulated_samples=50)


class TestValidationSchedulerInitialization:
    def test_initial_state_is_zero(self):
        vs = ValidationScheduler(
            validation_frequency=5,
            validate_at_epoch=True,
            min_samples_per_validation=100,
            validate_at_final=True,
        )
        assert vs.last_validated_samples == 0
        assert vs.last_sample_based_validation_samples == 0
        assert vs.last_epoch_validated_samples == 0
        assert vs.last_final_validated_samples == 0

    def test_all_triggers_configured(self):
        vs = ValidationScheduler(
            validation_frequency=10,
            validate_at_epoch=True,
            min_samples_per_validation=500,
            validate_at_final=True,
        )
        assert vs.is_configured
        assert vs.validation_frequency == 10
        assert vs.validate_at_epoch is True
        assert vs.min_samples_per_validation == 500
        assert vs.validate_at_final is True
