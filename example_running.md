# DASD Command Line Examples

## Training

- Train with default settings (1500 steps)

	```shell
	python run.py train
	```

- Train with custom settings

	```shell
	python run.py train --max_steps 2000 --output_dir my_output
	```

## Evaluation

- Evaluate trained model against baseline SD

	```shell
	python run.py evaluate
	```

- Evaluate from specific checkpoint

	```shell
	python run.py evaluate --checkpoint path/to/checkpoint
	```

## Generation

- with explicit domain tokens as examples

	```shell
	python run.py generate --domain satellite --num_images 8
	python run.py generate --domain xray --num_images 4
	```

- with custom prompt

	```shell
	python run.py generate --domain xray --prompt "chest scan showing healthy lungs"
	```

- with automatic domain detection

	```shell
	python run.py generate --auto-domain --prompt "chest scan showing lungs"
	python run.py generate --auto-domain --num_images 4
	```


## FULL PIPELINE

- train + evaluate + generate

	```shell
	python run.py all
	```

- add custom settings

	````shell
	python run.py all --max_steps 1500 --num_images 8
	````



# Python API Examples (see API.py for full details)
```shell
python API.py train      # Train the model
python API.py generate   # Generate with explicit tokens
python API.py auto       # Generate with auto-domain
python API.py classify   # Classify prompts
python API.py batch      # Batch generation
python API.py compare    # Domain comparison
```

