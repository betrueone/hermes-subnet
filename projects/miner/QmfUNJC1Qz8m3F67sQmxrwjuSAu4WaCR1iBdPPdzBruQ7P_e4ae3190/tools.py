import aiohttp
from langchain_core.tools import tool
from typing import Any, Dict

async def request_subquery(options: Dict[str, Any]):
    async with aiohttp.ClientSession() as session:
        payload = {
            "variables": options.get("variables", {}),
             "query": options["query"]
        }
        url = options.get("url") or "https://index-api.onfinality.io/sq/subquery/subquery-mainnet"
        timeout = options.get("timeout", 30)
        method = options.get("method", "POST").upper()

        async with session.request(
            method,
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as resp:
            result = await resp.json()
            res = result.get("data", {}).get(options["type"])
        return res

@tool
async def query_indexer_rewards(indexer: str, era: str) -> int:
    """
    Query the total rewards for a specific indexer in a given era.
     
    Do NOT call this tool when:
        1. The query is related to Stake, APY, Commission Rate or other non-reward metrics.

    Args:
        indexer (str): The indexer address or identifier
        era (str): Era number in two supported formats:
                  - Hexadecimal format: e.g., "0x48"
                  - Decimal format: e.g., "72" (equivalent to 0x48)
    
    Returns:
        int: Total rewards earned by the indexer in the specified era,
             returned in 18-decimal precision SQT (wei units).
             Can be converted to ETH if needed (1 ETH = 10^18 wei).
    
    Examples:
        - query_indexer_rewards("indexer_address", "0x48")
        - query_indexer_rewards("indexer_address", "72")
    """

    query = '''
    query (
      $id: String!
    ) {
      indexerReward(
        id: $id
      ) {
        id
        amount
      }
    }
    '''

    if era.startswith("0x"):
        era_hex = era.lower()
    else:
        try:
            era_hex = hex(int(era))
        except Exception:
            era_hex = era

    r = await request_subquery({
        "query": query,
        "type": "indexerReward",
        "variables": {
            "id": f"{indexer}:{era_hex}"
        },
    })
    return r.get('amount') if r else 0

tools = [query_indexer_rewards]