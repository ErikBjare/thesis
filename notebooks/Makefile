.PHONY: build

build: output/Main.html output/Signal.html output/Activity.html

output/%.html: %.ipynb
	poetry run jupyter nbconvert $^ --output-dir output --to=html --execute --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=600

precommit:
	poetry run jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace *.ipynb

clean:
	rm -r output
