"""Old NeuS Hybrid protocol with Shell optimized throughout all stages."""
from hairgs_hybrid import surface_prior_hybrid_v1 as base
from hairgs_hybrid.teacher_free_shell_strand_v1.training import Schedule


base.SCHEDULE = Schedule(stage1024=1000, stage2048=3000, total_steps=8000,
                         max_shell_splits=0, persistent_shell=True)


if __name__ == "__main__":
    base.main()
