"""Run the rl_meta_pose_feedback replay entry point."""
if __package__:
    from ._bootstrap import activate
else:
    from _bootstrap import activate
activate()
from rl_meta_pose_feedback.visualization.replay import main

if __name__ == '__main__':
    main()
