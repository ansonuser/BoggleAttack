from boggle.attacks.helper.playing import a_simulate
from boggle.utils import callback
import asyncio
import nest_asyncio
from configs.sentence import targets
import pandas as pd
import click


@click.command()
@click.option('--mode', type=click.Choice(['tree', 'liner']), default='tree', help='Mode of operation: tree or liner')
def main(mode):
    loop = asyncio.get_event_loop()
    if loop.is_running():
        nest_asyncio.apply()

    mode_tree = (mode == 'tree')
    out = loop.run_until_complete(a_simulate(targets, callback, mode_tree=mode_tree))

    if mode_tree:
        df = pd.DataFrame(out, columns=["target", "best_response", "best_attack_prompt", "best_score", "conversation_history"])
        df.to_excel("attack_result_tree_bulk.xlsx", index=False)
    else:
        df = pd.DataFrame(out, columns=["rephrased_text", "last_response", "final_attack_prompt", "refused", "num_of_rounds", "conversation_history"])
        df["target"] = targets
        df = df[["target", "rephrased_text", "last_response", "final_attack_prompt", "refused", "num_of_rounds", "conversation_history"]]
        df.to_excel("attack_result_liner.xlsx", index=False)
    
    print(out)
  

if __name__ == "__main__":
    main()

