persuasion_deliberation_and_negotiation_domain_prompt = (
    """
This dialogue type focuses on **resolving conflicts of interest** or **reconciling differing viewpoints** to reach a mutually acceptable agreement. Participants engage in **reason-based proposals** and **trade-offs**, aiming for practical, mutually beneficial outcomes.

- **Primary Goals**:
  - Convince or compromise with others using logic and evidence.
  - Resolve conflicts by making offers and concessions.
  - Secure a final agreement that addresses conflicting interests.

- **Typical Moves**:
  - Proposing clear offers with conditions (“If you accept X, I’ll agree to Y”).
  - Negotiating with counteroffers (“That won’t work, but I can propose Z instead”).
  - Emphasizing shared goals and summarizing priorities.

- **Style**:
  - **Collaborative but strategic**, with a focus on practical outcomes and logical proposals.
  - Avoids personal attacks and highlights benefits or trade-offs for each side.

**Key Indicators**:
- Iterative **offer–counteroffer patterns** with explicit conditions.
- Efforts to resolve differing interests and achieve practical outcomes.
- Dialogue often concludes with **an agreement or resolved conflict**.
""",
)

inquiry_and_information_seeking_domain_prompt = (
    """
This dialogue type revolves around **exploring unknowns** and **filling knowledge gaps**. Participants aim to learn, clarify, or confirm information through structured exchanges that emphasize **knowledge exchange** and **fact verification**.

- **Primary Goals**:
  - Obtain accurate information or validate existing knowledge.
  - Clarify unclear concepts or explore new evidence.

- **Typical Moves**:
  - Asking specific, focused questions (“Where does this data come from?” “What does this term mean?”).
  - Requesting sources, elaborations, or examples.
  - Testing the reliability and validity of the information provided.

- **Style**:
  - **Inquisitive and neutral**, with logical follow-ups to maintain clarity.
  - Participants may withhold judgments or opinions unless necessary.

**Key Indicators**:
- Frequent **question–answer patterns** focusing on facts and sources.
- Absence of offers or trade-offs, focusing entirely on **learning and understanding**.
- Ends when **knowledge is clarified or confirmed**, not when agreements are reached.

""",
)

eristic_domain_prompt = """
An **Eristic** dialogue arises from **antagonism** or **hostility**, focusing on **winning** an argument or **dominating** an opponent. Participants aim to **attack, undermine, or outmaneuver** each other’s positions rather than seeking truth or consensus. Emotional appeals, personal attacks, and **point‑scoring** are common.

- **Primary Goals**: Achieve **victory** in a debate; maintain or bolster personal prestige; sometimes simply vent or amuse oneself by defeating the opposition.
- **Typical Moves**:
  - Accusing, insulting, or belittling the other side.
  - Using sarcasm, ridicule, or straw‑man arguments.
  - Shifting the topic or using fallacies to maintain an advantage.
  - Exaggerating flaws in the opponent’s logic to sway onlookers.
- **Secondary Goals**: Gain experience in debate, gain social status, or entertain an audience.
- **Style**: Confrontational, emotionally charged, often less structured or cooperative. Participants **rarely** make concessions or aim for compromise.

**Key Indicators**  
- Heightened emotional language (“That’s absurd,” “You clearly have no idea…”).
- Frequent interruptions or dismissive retorts.
- Focus on personal victory over mutual understanding.
"""

domain_prompt_dict = {
    "Persuassion_Deliberation_and_Negotiation": persuasion_deliberation_and_negotiation_domain_prompt,
    "Inquiry_and_Information_Seeking": inquiry_and_information_seeking_domain_prompt,
    "Eristic": eristic_domain_prompt,
}
