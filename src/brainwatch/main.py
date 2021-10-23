import click

# TODO: Move contents into brainwatch module
import eegwatch.main
import eegclassify.main


@click.group()
def main():
    pass


main.add_command(eegwatch.main.main)
main.add_command(eegclassify.main.main)

if __name__ == "__main__":
    main()
