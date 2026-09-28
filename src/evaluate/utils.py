import statistics

# Full ALE action space; index == raw ALE action ID, identical for every game.
ALE_ACTION_NAMES = (
    "NOOP",
    "FIRE",
    "UP",
    "RIGHT",
    "LEFT",
    "DOWN",
    "UPRIGHT",
    "UPLEFT",
    "DOWNRIGHT",
    "DOWNLEFT",
    "UPFIRE",
    "RIGHTFIRE",
    "LEFTFIRE",
    "DOWNFIRE",
    "UPRIGHTFIRE",
    "UPLEFTFIRE",
    "DOWNRIGHTFIRE",
    "DOWNLEFTFIRE",
)


def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_action_usage(counts, total: int, indent: str = "") -> None:
    for action_id, name in enumerate(ALE_ACTION_NAMES):
        count = counts[action_id]
        pct = 100.0 * count / total if total > 0 else 0.0
        print(f"{indent}Action {action_id:2d} ({name:14s}): {count:6d} ({pct:6.2f}%)")


def print_step(step: int, action: int, reward: float) -> None:
    print(
        f"Step {step:4d} | Action {action:2d} "
        f"({ALE_ACTION_NAMES[action]:14s}) | Reward {reward:7.2f}"
    )


def print_episode_report(
    episode: int,
    step_count: int,
    total_reward: float,
    action_counts,
    final_info: dict,
    truncated_by_max_steps: bool,
    max_steps: int,
) -> None:
    """Per-episode summary; episode is 1-based."""
    print(f"\nEpisode {episode} finished")
    print("Steps:", step_count)
    print("Total reward:", total_reward)

    if "score" in final_info:
        print("Final score:", final_info["score"])

    print("\nAction usage:")
    print_action_usage(action_counts, step_count, indent="  ")

    if truncated_by_max_steps:
        print(f"Episode stopped because it reached gym_max_steps={max_steps}.")


def print_summary_report(episode_rewards, total_action_counts) -> None:
    """Overall reward and action-usage summary across all episodes."""
    print_header("EVALUATION SUMMARY")
    print(f"Evaluated {len(episode_rewards)} episodes")
    print("Per-episode rewards:", episode_rewards)
    print(
        f"Mean reward: {statistics.mean(episode_rewards):.2f} "
        f"(std: {statistics.pstdev(episode_rewards):.2f})"
    )

    total_actions = sum(total_action_counts.values())
    print_header("TOTAL ACTION USAGE")
    print(f"Total actions executed: {total_actions}")
    print_action_usage(total_action_counts, total_actions)
