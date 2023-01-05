import typer
from rich import inspect
from rich import print as pprint
from rich.console import Console
from typer import Option

from mmseg.utils import collect_env

console = Console()


def name_callback(ctx: typer.Context, value: str):
    inspect(ctx, methods=True)
    inspect(value)
    if ctx.resilient_parsing:
        return
    print("Validating name")
    if value != "Camila":
        raise typer.BadParameter("Only Camila is allowed")
    return value


def main(
    seed: int = Option(0, help="random seed"),
    name: str = Option("Camila", help="this is name", callback=name_callback),
):
    console.log(log_locals=True)

    env_info_dict = collect_env()
    env_info = "\n".join([f"{k}: {v}" for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    print("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    pprint(env_info_dict)
    inspect(env_info_dict)
    # inspect(dataclass(**env_info_dict))

    # renderables = self._collect_renderables(
    #     objects,
    #     sep,
    #     end,
    #     justify=justify,
    #     emoji=emoji,
    #     markup=markup,
    #     highlight=highlight,
    # )
    # if style is not None:
    #     renderables = [Styled(renderable, style) for renderable in renderables]

    # renderables.append(render_scope(env_info_dict, title="[i]locals"))

    return


if __name__ == "__main__":
    typer.run(main)
