# Ganglion: A Substrate for Autonomous Agentic Mining on Bittensor

**Infrastructure for autonomous miners that research and self-improve.**

---

## What Mining on Bittensor Actually Is

Mining on Bittensor is as diverse as the subnets themselves. Some miners run fine-tuned models. Some run inference pipelines. Some do scientific compute, market forecasting, data labelling, protein folding, code generation. The substrate varies enormously — but the challenge every serious miner faces converges on the same thing: **how do you keep improving faster than the subnet changes and faster than your competitors adapt?**

Bittensor subnets are living systems. Validators update scoring logic. Incentive landscapes shift as competing miners improve. A strategy that earns strong emissions today can degrade to nothing in weeks without anyone changing a line of code. The subnet moved.

The miners who sustain performance aren't the ones who found the best initial strategy — they're the ones who built the best process for continuously discovering better ones. Every submission is an experiment. Every score is a signal. Every failure is information. Mining isn't a deployment. It's a development cycle.

Ganglion is the infrastructure for that cycle.

---

## The Idea

Nature doesn't encode intelligence. It creates conditions for intelligence to emerge.

Ant colonies solve complex routing problems without any ant understanding the problem. Immune systems defeat threats never previously encountered. Brains consolidate experience into judgment without anyone deciding what to remember.

The mechanism in every case is the same: simple rules, applied consistently, compound into complex adaptive behaviour over time.

Ganglion applies this to Bittensor mining. Instead of encoding mining strategies, it creates the conditions for a miner to discover and continuously refine its own — through structured memory, collective learning, and self-directed experimentation.

---

## Five Rules. Everything Else Emerges.

**1. Every action leaves a trace.**
Every stage run, every agent attempt, every pipeline execution writes a typed record to the knowledge store. Always. Unconditionally.

**2. Every decision reads the trace.**
Before any stage runs, relevant history is injected into the prompt. The miner doesn't start from scratch each cycle — it starts from everything it has already learned.

**3. Structure is declared, not decided.**
Routing, retry behaviour, compute allocation are configuration. Agents decide *what* to try. Infrastructure decides *where* and *how* to run it. These are different concerns and they stay in different layers.

**4. Interfaces are narrow, implementations are free.**
Every major component is a protocol with the minimum possible surface area. Swap backends, policies, and agents without touching orchestration.

**5. Agents are peers, not hierarchy.**
No agent coordinates another. Collective intelligence emerges from many agents independently following rules 1 and 2 against shared state — the same way ant colonies encode environmental knowledge without any ant knowing the colony's strategy.

---

## What This Produces

A miner on its first run is not much smarter than a naive loop.

After fifty runs it has accumulated a detailed picture of what works on its target subnet — which model families score best, which calibration approaches are reliable, which pipeline shapes fail and where. That picture lives in the knowledge store and gets injected into every subsequent decision.

After a hundred runs across a fleet of agents, each independently exploring and writing back what it finds, that picture is richer than any single human researcher could build manually. Confirmed patterns — strategies validated by multiple independent agents — surface first. Dead ends stay buried.

The miner isn't getting smarter because someone improved the code. It's getting smarter because every run makes the next run better informed. That's the compounding effect. It doesn't require intervention. It just requires the five rules to hold.

---

## The Memory Model

Ganglion treats memory the way the brain does — not as static storage but as a dynamic, adaptive process.

**Patterns** (what worked) and **antipatterns** (what failed) accumulate after every stage. High-frequency, high-value signals get reinforced — strategies confirmed by multiple agents carry more weight in injection than strategies seen once. This is Hebbian plasticity applied to mining knowledge: connections that fire together, wire together.

**Agent trust** is derived from each agent's recent history — not a separate system, just a view over the same data. An agent with a strong track record has more influence on the shared pool. An agent that's consistently failing has less, until it finds its footing.

**Cross-agent transfer** means one agent's discovery becomes another agent's starting point. An explorer agent running broad searches on cheap cloud GPUs writes what it finds. An optimizer agent running deep refinements on a local rig reads those findings and exploits the best leads. Neither coordinates the other. The shared knowledge store is the environment — and the environment is the memory.

---

## Infrastructure, Not Strategy

Ganglion doesn't know anything about Bittensor subnets. It doesn't know what a good CRPS score looks like or which model families work for price forecasting. It doesn't know your subnet's scoring function or what hardware your experiments need. It provides the substrate for a miner to figure all of that out itself — and keep figuring it out as the subnet changes.

The compute layer routes jobs to the right hardware without the agent touching credentials. The retry layer escalates to better models or higher temperatures without the orchestrator making decisions. The knowledge layer injects relevant history without the agent knowing it's there.

Every layer does one thing. The intelligence lives in the compounding, not the components.

---

## Getting Started

```bash
git clone https://github.com/TensorLink-AI/ganglion.git
cd ganglion
pip install -e ".[dev]"
```

The `synth-city` example is a fully configured autonomous miner for Bittensor SN50 — probabilistic price forecasting, scored on CRPS across 13 assets. It shows the full development cycle: deterministic tool stages for data fetching and scoring, LLM reasoning stages for planning and calibration, knowledge store accumulation, compute routing, and retry policies.

```bash
cd examples/synth-city
python -c "from config import pipeline; print(pipeline.validate() or 'Valid')"
```

Scaffold a new subnet project:

```bash
python -m ganglion init --subnet 42 --name my-miner
```

---

## Contributing

The five rules above are the design review checklist. Before proposing a change, ask: does this serve one of them? Does this encode specific behaviour that should emerge instead? Is this solving a problem we have evidence of, or one we're anticipating?

If removing the proposed addition wouldn't break any of the five rules, it probably shouldn't be added yet.

Complexity that doesn't serve emergence is just complexity.
