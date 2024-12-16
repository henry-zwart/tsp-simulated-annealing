FIGURES_DIR = results/figures
DATA_DIR = data

COOLING = linear exponential inverse_log
FIGURE_NAMES = \
	chain_length_error.pdf \
	$(patsubst %, %_trace.pdf, $(COOLING)) \
	$(patsubst %, %_dist.pdf, $(COOLING))

FIGURES = $(patsubst %, results/figures/%, $(FIGURE_NAMES))

ENTRYPOINT ?= uv run

all: results/plot_metadata.json results/experiment_metadata.json

results/plot_metadata.json: experiments/plots.py results/experiment_metadata.json | $(FIGURES_DIR)
	$(ENTRYPOINT) $<

$(FIGURES_DIR):
	mkdir -p $@

results/experiment_metadata.json: \
			scripts/combine_metadata.py \
			data/chain_length_error.meta \
			data/markov_chains.meta \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $<

data/%.meta: experiments/%.py | $(DATA_DIR)
	$(ENTRYPOINT) $<


$(DATA_DIR):
	mkdir -p $@

.PHONY: clean
clean:
	rm -rf results data
