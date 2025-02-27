# mtg_tool.py
import httpx
from typing import Any, Dict
from langchain_core.tools import tool

@tool
def mtg_search(params: Dict[str, Any]) -> str:
    """
    Searches for Magic: The Gathering cards using the magicthegathering.io API.
    
    The `params` dict can include any of the supported query parameters,
    such as:
      - name: str
      - set: str
      - types: str (comma-separated list)
      - subtypes: str
      - colors: str (comma-separated list)
      - rarity: str
      - cmc: int or str
      - manaCost: str
      etc.
    
    Returns:
        A formatted string with details about each matching card.
    """
    url = "https://api.magicthegathering.io/v1/cards"
    response = httpx.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    cards = data.get("cards", [])
    if cards:
        result_lines = []
        for card in cards:
            name = card.get("name", "Unknown")
            set_name = card.get("setName", "Unknown Set")
            mana_cost = card.get("manaCost", "N/A")
            cmc = card.get("cmc", "N/A")
            colors = card.get("colors", [])
            types = card.get("types", [])
            text = card.get("text", "No text provided")
            power = card.get("power", "N/A")
            toughness = card.get("toughness", "N/A")
            rarity = card.get("rarity", "N/A")
            flavor = card.get("flavor", "No flavor text")
            formatted = (
                f"Name: {name}\n"
                f"Set: {set_name}\n"
                f"Mana Cost: {mana_cost}\n"
                f"CMC: {cmc}\n"
                f"Colors: {', '.join(colors) if colors else 'None'}\n"
                f"Types: {', '.join(types) if types else 'None'}\n"
                f"Text: {text}\n"
                f"Power/Toughness: {power}/{toughness}\n"
                f"Rarity: {rarity}\n"
                f"Flavor: {flavor}\n"
            )
            result_lines.append(formatted)
        return "\n".join(result_lines)
    else:
        return "No cards found."