from pathlib import Path

import transformers


def load_last_ckpt(ckpt_dir, checkpoint_id=None):
    """Load the last checkpoint from a directory.

    Parameters
    ----------
    ckpt_dir : str | Path
        Path to the directory where the checkpoints are saved.
    checkpoint_id : int
        The checkpoint id to load. If None, the last checkpoint is loaded.
    """
    ckpt_dir = Path(ckpt_dir)

    if checkpoint_id is None:
        checkpoint_id = sorted([
            int(p.name.replace("checkpoint-", ""))
            for p in ckpt_dir.glob("checkpoint-*")
        ])[-1]

    last_checkpoint = f"checkpoint-{checkpoint_id}"
    return transformers.GPTNeoForCausalLM.from_pretrained(
        ckpt_dir / last_checkpoint
    )
