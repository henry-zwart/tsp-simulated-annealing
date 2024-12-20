FIGURES_DIR = results/figures
DATA_DIR = data

COOLING = linear exponential inverse_log
CHAIN_LENGTHS = 200 1500
FIGURE_NAMES = \
	chain_length_error.pdf \
	$(patsubst %, %_trace.pdf, $(COOLING)) \
	$(patsubst %, %_dist.pdf, $(COOLING))

FIGURES = $(patsubst %, results/figures/%, $(FIGURE_NAMES))

ENTRYPOINT ?= uv run

all: results/plot_metadata.json results/experiment_metadata.json data/cooling.meta

results/plot_metadata.json: experiments/plots.py results/experiment_metadata.json | $(FIGURES_DIR)
	$(ENTRYPOINT) $< $(CHAIN_LENGTHS)

$(FIGURES_DIR):
	mkdir -p $@


results/tests.json: experimens/stat_tests.py data/cooling.meta | $(FIGURES_DIR)
	$(ENTRYPOINT) $<


results/experiment_metadata.json: \
			scripts/combine_metadata.py \
			data/chain_length_error.meta \
			$(patsubst %, data/markov_chains_%.meta, $(CHAIN_LENGTHS)) \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $<


data/cooling.meta: \
			experiments/parameterisation_data_prep.py \
			experiments/error_CS_data_prep.py \
			experiments/plot_cooling_schedules.py \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) experiments/parameterisation_data_prep.py && \
	$(ENTRYPOINT) experiments/error_CS_data_prep.py && \
	$(ENTRYPOINT) experiments/plot_cooling_schedules.py


data/markov_chains_%.meta: experiments/markov_chains.py | $(DATA_DIR)
	$(ENTRYPOINT) $< $*

data/%.meta: experiments/%.py | $(DATA_DIR)
	$(ENTRYPOINT) $<



$(DATA_DIR):
	mkdir -p $@

.PHONY: clean
clean:
	rm -rf results data
