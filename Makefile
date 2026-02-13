include $(ENV_FILE)
export

PYTHONPATH=./

metagraph:
	btcli subnet metagraph --netuid $(NETUID) --subtensor.chain_endpoint $($(SUBTENSOR_NETWORK))

miner:
	pm2 start neurons/miner.py --name $(MINER_NAME) --interpreter .venv/bin/python -- \
		--neuron.name $(MINER_NAME) \
		--wallet.name $(WALLET_NAME) \
		--wallet.hotkey $(HOTKEY) \
		--axon.port $(PORT) \
		--netuid $(NETUID)

miner_dev:
	python neurons/miner.py \
		--neuron.name $(MINER_NAME) \
		--wallet.name $(WALLET_NAME) \
		--wallet.hotkey $(HOTKEY) \
		--axon.port $(PORT) \
		--netuid $(NETUID)