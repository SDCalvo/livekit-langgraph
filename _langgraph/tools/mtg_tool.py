from typing import Optional, Union
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import httpx
import logging

logger = logging.getLogger(__name__)

class MTGSearchInput(BaseModel):
    """
    Schema for Magic: The Gathering card search.
    """
    name: Optional[str] = Field(
        default=None,
        description="The card name (partial or full) to search for. For an exact match, enclose the name in double quotes."
    )
    set: Optional[str] = Field(
        default=None,
        description="The set code or set name to filter the cards."
    )
    types: Optional[str] = Field(
        default=None,
        description="A comma- or pipe-separated list of card types (e.g., Creature, Sorcery)."
    )
    colors: Optional[str] = Field(
        default=None,
        description="A comma- or pipe-separated list of card colors (e.g., 'W,U' or 'W|U')."
    )
    rarity: Optional[str] = Field(
        default=None,
        description="The rarity of the card (e.g., Common, Uncommon, Rare, Mythic Rare)."
    )
    cmc: Optional[Union[int, str]] = Field(
        default=None,
        description="Converted mana cost (an integer or a string, for special cases)."
    )
    # You can add more fields if needed (like types, subtypes, etc.)

@tool(
    args_schema=MTGSearchInput,
    description=(
        "Searches for Magic: The Gathering cards using the magicthegathering.io API. "
        "Supported filters include name, set, types, colors, rarity, and cmc. "
        "Returns details for all matching cards."
    ),
    response_format="content"
)
def mtg_search(
    name: Optional[str] = None,
    set: Optional[str] = None,
    types: Optional[str] = None,
    colors: Optional[str] = None,
    rarity: Optional[str] = None,
    cmc: Optional[Union[int, str]] = None
) -> str:
    """
    Searches for Magic: The Gathering cards using the magicthegathering.io API.
    
    The parameters should conform to the MTGSearchInput schema.
    
    Returns:
        A formatted string with details about each matching card.
    """
    url = "https://api.magicthegathering.io/v1/cards"

    params = {}
    if name:
        params["name"] = name
    if set:
        params["set"] = set
    if types:
        params["types"] = types
    if colors:
        params["colors"] = colors
    if rarity:
        params["rarity"] = rarity
    if cmc:
        params["cmc"] = cmc

    try:
        response = httpx.get(url, params=params)
        response.raise_for_status()
    except Exception as e:
        error_msg = f"Error calling MTG API: {str(e)}"
        logger.error(error_msg)
        return error_msg

    data = response.json()
    cards = data.get("cards", [])
    if not cards:
        return "No cards found."

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
            "----------------------------------"
        )
        result_lines.append(formatted)
    return "\n".join(result_lines)