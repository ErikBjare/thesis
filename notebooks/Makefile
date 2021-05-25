.PHONY: build

build: output/Main.html output/Signal.html output/Activity.html

output/%.html: %.ipynb
	poetry run jupyter nbconvert $^ --output-dir output --to=html --execute --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=600

precommit:
	poetry run jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace *.ipynb

set-kernel:
	sed -i 's/"name": "erb-thesis"/"name": "python3"/g' *.ipynb

clean:
	rm -r output
