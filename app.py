import json
import math
import random
import re
from collections import Counter
from pathlib import Path

from flask import Flask, render_template, request, jsonify, session

app = Flask(__name__)
app.config["SECRET_KEY"] = "a9-dev-secret-key"

# ---------------- DATASET ----------------
KNOWLEDGE_BASE_PATH = Path(__file__).with_name("knowledge_base.json")

with KNOWLEDGE_BASE_PATH.open("r", encoding="utf-8") as f:
    KNOWLEDGE_BASE = json.load(f)

TOPIC_KEYWORDS = {
    "skills": {
        "skill", "skills", "expertise", "good", "proficient", "tools", "tech", "technologies",
        "strength", "strengths", "qualified", "qualification", "qualifications", "competency",
        "competencies", "technical", "soft", "stack", "capable"
    },
    "education": {
        "education", "school", "college", "university", "degree", "degrees", "major", "study", "studying",
        "academic", "academics", "coursework", "graduate", "graduated", "certification", "certifications",
        "pursuing", "enrolled", "attended", "graduated", "diploma", "bachelor", "master", "associate"
    },
    "experience": {
        "experience", "job", "jobs", "career", "background",
        "responsibility", "responsibilities", "duty", "duties", "achievement", "achievements",
        "accomplishment", "accomplishments", "managed", "led", "history", "prior", "previously"
    },
    "employer": {
        "company", "employer", "employed", "organization", "works", "working", "workplace", "work", "worked"
    },
    "headline": {
        "headline", "title", "position", "current", "role"
    },
    "location": {
        "location", "where", "city", "from", "based", "area", "live", "lives", "located"
    },
    "about": {
        "about", "summary", "introduce", "introduction", "overview", "interests",
        "interest", "goal", "goals", "motivation", "motivated", "why", "hire", "fit",
        "tell", "describe", "background", "profile", "profiles", "summarize"
    },
    "availability": {
        "availability", "available", "start", "starting", "notice"
    },
    "preferred_roles": {
        "preferred", "preference", "interested", "interest", "seeking", "seeks", "looking", "target", "roles"
    },
    "remote_preference": {
        "remote", "hybrid", "onsite", "on-site", "site", "wfh", "office", "in-person", "open"
    },
    "relocation": {
        "relocation", "relocate", "moving", "move"
    },
    "contact": {
        "contact", "reach", "linkedin", "profile"
    },
    "compensation": {
        "salary", "compensation", "pay", "range", "bonus"
    },
    "authorization": {
        "visa", "sponsorship", "authorized", "authorization", "workauth", "citizenship"
    },
}

STOP_WORDS = {
    "what", "is", "are", "does", "do", "the", "a", "an", "his", "her", "their", "can", "you",
    "give", "some", "info", "on", "me", "tell", "please", "to", "of"
}

PRONOUN_HINTS = {
    "he", "she", "his", "her", "they", "them", "their", "him"
}

FOLLOW_UP_CONTEXT_HINTS = {
    "also", "too", "about", "and"
}

ABBREVIATIONS = {
    r"\bneiu\b": "northeastern illinois university",
    r"\bcs\b": "computer science",
    r"\bui\b": "user interface",
    r"\bux\b": "user experience",
    r"\bml\b": "machine learning",
    r"\bai\b": "artificial intelligence",
    r"\bdb\b": "database",
    r"\bjs\b": "javascript",
    r"\bts\b": "typescript",
    r"\bsql\b": "sql",
    r"\bapi\b": "api",
}

ABBREVIATION_PATTERNS = [(re.compile(pattern), expansion) for pattern, expansion in ABBREVIATIONS.items()]


def _expand_abbreviations(text):
    result = text.lower()
    for pattern, expansion in ABBREVIATION_PATTERNS:
        result = pattern.sub(expansion, result)
    return result


def _tokenize(text):
    return [t for t in re.findall(r"\b\w+\b", text.lower()) if t not in STOP_WORDS]


def _build_chunks(profiles):
    chunks = []

    for person in profiles:
        name = person.get("name", "Unknown")
        headline = person.get("headline", "")
        location = person.get("location", "")
        about = person.get("about", "")

        if about:
            chunks.append({
                "person": name,
                "topic": "about",
                "text": f"{name}. {headline}. Location: {location}. {about}".strip()
            })

        for edu in person.get("education", []):
            chunks.append({
                "person": name,
                "topic": "education",
                "text": (
                    f"{name} education: {edu.get('degree', 'Degree')} in "
                    f"{edu.get('field', 'N/A')} at {edu.get('school', 'N/A')}."
                )
            })

        for exp in person.get("experience", []):
            chunks.append({
                "person": name,
                "topic": "experience",
                "text": (
                    f"{name} experience: {exp.get('title', 'Role')} at {exp.get('company', 'N/A')}. "
                    f"{exp.get('description', '')}"
                ).strip()
            })

        skills = person.get("skills", [])
        if skills:
            chunks.append({
                "person": name,
                "topic": "skills",
                "text": f"{name} skills: {', '.join(skills)}."
            })

        preferred_roles = person.get("preferred_roles", [])
        if preferred_roles:
            chunks.append({
                "person": name,
                "topic": "preferred_roles",
                "text": f"{name} preferred roles: {', '.join(preferred_roles)}."
            })

        availability = person.get("availability")
        if availability:
            chunks.append({
                "person": name,
                "topic": "availability",
                "text": f"{name} availability: {availability}."
            })

        remote_preference = person.get("remote_preference")
        if remote_preference:
            chunks.append({
                "person": name,
                "topic": "remote_preference",
                "text": f"{name} remote preference: {remote_preference}."
            })

        relocation = person.get("relocation")
        if relocation:
            chunks.append({
                "person": name,
                "topic": "relocation",
                "text": f"{name} relocation preference: {relocation}."
            })

    return chunks


def _build_idf(chunks):
    doc_freq = Counter()
    total_docs = max(1, len(chunks))

    for chunk in chunks:
        terms = set(_tokenize(chunk["text"]))
        for term in terms:
            doc_freq[term] += 1

    idf = {}
    for term, df in doc_freq.items():
        idf[term] = math.log((total_docs + 1) / (df + 1)) + 1.0

    return idf


def _tfidf_vector(text, idf):
    tokens = _tokenize(text)
    if not tokens:
        return {}

    tf = Counter(tokens)
    total = float(len(tokens))
    vec = {}
    for term, count in tf.items():
        weight = (count / total) * idf.get(term, 0.0)
        if weight > 0:
            vec[term] = weight
    return vec


def _cosine_similarity(vec_a, vec_b):
    if not vec_a or not vec_b:
        return 0.0

    shared = set(vec_a.keys()) & set(vec_b.keys())
    dot = sum(vec_a[t] * vec_b[t] for t in shared)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


CHUNKS = _build_chunks(KNOWLEDGE_BASE)
IDF = _build_idf(CHUNKS)
CHUNK_VECTORS = [_tfidf_vector(chunk["text"], IDF) for chunk in CHUNKS]

# O(1) person lookup by full name
PERSON_BY_NAME = {p.get("name", ""): p for p in KNOWLEDGE_BASE}

# Precomputed lowercase JSON strings for fast literal scan in broad queries
PERSON_JSON_CACHE = {p.get("name", ""): json.dumps(p).lower() for p in KNOWLEDGE_BASE}

# Precomputed name/nickname tokens for fast person matching
PERSON_QUERY_TOKENS = {}
for person in KNOWLEDGE_BASE:
    person_name = person.get("name", "")
    person_tokens = set(_tokenize(person_name))
    for nickname in person.get("nicknames", []):
        person_tokens |= set(_tokenize(nickname))
    PERSON_QUERY_TOKENS[person_name] = person_tokens


def _find_person(tokens):
    best_person = None
    best_score = 0

    for person_name, person_tokens in PERSON_QUERY_TOKENS.items():
        score = len(tokens & person_tokens)
        if score > best_score:
            best_score = score
            best_person = PERSON_BY_NAME.get(person_name)

    return best_person if best_score > 0 else None


def _find_topic(tokens):
    best_topic = None
    best_score = 0

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = len(tokens & keywords)
        if score > best_score:
            best_score = score
            best_topic = topic

    return best_topic if best_score > 0 else None


# Generic conversation handlers
GREETINGS = {
    "hello", "hi", "hey", "greetings", "howdy", "what's up", "whats up", "yo", "sup"
}

OFF_TOPIC_KEYWORDS = {
    "weather", "temperature", "rain", "snow", "forecast", "climate",
    "sports", "game", "score", "football", "basketball", "soccer",
    "news", "politics", "election", "covid", "pandemic",
    "recipe", "cook", "food", "restaurant",
    "movie", "film", "actor", "actress", "celebrity",
    "joke", "funny", "laugh", "meme",
    "bitcoin", "crypto", "stock", "market",
    "math", "physics", "chemistry", "biology",
    "calculate", "plus", "minus", "times", "equals", "divide", "multiply",
    "equation",
}

MIN_RETRIEVAL_CONFIDENCE = 0.35


def _is_math_query(raw_input):
    """Check if input contains math operators — but not tech names like C++, C#, .NET"""
    # Remove known tech-name patterns before checking for math operators
    cleaned = re.sub(r'\b[a-zA-Z][a-zA-Z0-9]*[+#]+\b', '', raw_input)  # C++, C#, A+
    cleaned = re.sub(r'\.NET\b', '', cleaned, flags=re.IGNORECASE)
    return any(op in cleaned for op in ["+", "-", "*", "/", "=", "^"])


def _extract_tech_names(text):
    return re.findall(r'\b[a-zA-Z][\w.]*[+#]+\b|\.NET\b|\bNode\.js\b|\bASP\.NET\b', text, re.IGNORECASE)


_RESPONSES_OFF_TOPIC = [
    "I'm specialized in team information. I can't help with that, but I'd love to tell you about Aaron, Rubi, Ezza, Rasheed, or Albion instead.",
    "That's outside my expertise. I focus on our team members. Ask me about Aaron, Rubi, Ezza, Rasheed, or Albion.",
    "Interesting question, but I'm here for team info. Let me tell you about one of our team members instead."
]
_RESPONSES_GREETING = [
    "Hey there. I'm Bob, your team knowledge assistant. My answers are based on the LinkedIn profiles of Aaron, Rubi, Ezza, Rasheed, and Albion. What would you like to know?",
    "Hello. I'm here to help you learn about our team. I pull info directly from their LinkedIn profiles — ask me anything about Aaron, Rubi, Ezza, Rasheed, or Albion.",
    "Hi. Welcome. I'm Bob. I'm powered by our team's LinkedIn profiles. Feel free to ask me about any of our 5 members."
]
_RESPONSES_HOW_ARE_YOU = [
    "I'm doing great, thanks for asking. I'm here to tell you about the team. Want to learn about Aaron, Rubi, Ezza, Rasheed, or Albion?",
    "I'm functioning perfectly. More importantly, I'd love to help you learn about our team members. Who would you like to know about?",
    "All systems go. I'm ready to share info about our team. Pick anyone: Aaron, Rubi, Ezza, Rasheed, or Albion."
]


def _handle_generic_input(user_input):
    tokens = set(_tokenize(user_input))

    if _is_math_query(user_input):
        return random.choice(_RESPONSES_OFF_TOPIC)

    if tokens & OFF_TOPIC_KEYWORDS:
        return random.choice(_RESPONSES_OFF_TOPIC)

    if tokens & GREETINGS:
        return random.choice(_RESPONSES_GREETING)

    if "how" in tokens and tokens & {"are", "you", "doing", "going", "feeling"}:
        return random.choice(_RESPONSES_HOW_ARE_YOU)

    return None


BROAD_QUERY_WORDS = {
    "who", "which", "all", "everyone", "each", "list", "members", "team", "anyone", "everyone", "whose", "someone",
    "group", "everybody", "both", "every", "member"
}

# Map TOPIC_KEYWORDS topics to actual chunk topics in CHUNKS
TOPIC_CHUNK_ALIAS = {
    "employer": "experience",
    "headline": "about",
}

BROAD_FILLER_WORDS = {
    "knows", "know", "use", "uses", "used", "using", "has", "have", "work", "works",
    "worked", "at", "in", "with", "for", "from", "attend", "attends", "attended",
    "studied", "study", "studies", "is", "was", "be", "been", "good", "skilled",
    "experienced", "proficient", "fluent", "familiar", "want", "wanted", "looking",
    "need", "needs", "find", "show", "tell", "give", "get"
}

def _has_topic_data(person, topic):
    if topic == "skills": return bool(person.get("skills"))
    if topic in ("experience", "employer"): return bool(person.get("experience"))
    if topic == "education": return bool(person.get("education"))
    if topic == "location": return bool(person.get("location"))
    if topic == "availability": return bool(person.get("availability"))
    if topic == "authorization": return bool(person.get("work_authorization"))
    if topic == "remote_preference": return bool(person.get("remote_preference"))
    if topic == "relocation": return bool(person.get("relocation"))
    if topic == "preferred_roles": return bool(person.get("preferred_roles"))
    if topic == "contact": return bool((person.get("contact") or {}).get("linkedin"))
    if topic == "about": return bool(person.get("about") or person.get("headline"))
    return True


# JSON field names that appear in every person's record — exclude from literal scan
JSON_KEY_WORDS = {
    "skills", "education", "experience", "name", "nicknames", "headline", "location",
    "about", "availability", "contact", "linkedin", "compensation", "authorization",
    "relocation", "remote", "preferred", "roles", "degree", "field", "school",
    "title", "company", "description", "null", "true", "false"
}


def _is_broad_query(tokens):
    return bool(tokens & BROAD_QUERY_WORDS)


def _find_all_people_for_topic(query, matched_topic, threshold=0.1):
    """Return all people whose topic chunks score above threshold for the query."""
    query_vec = _tfidf_vector(query, IDF)
    if not query_vec:
        return []
    person_best = {}

    for chunk, chunk_vec in zip(CHUNKS, CHUNK_VECTORS):
        if matched_topic and chunk.get("topic") != matched_topic:
            continue
        score = _cosine_similarity(query_vec, chunk_vec)
        person = chunk["person"]
        if score > person_best.get(person, 0):
            person_best[person] = score

    matching = [(score, PERSON_BY_NAME.get(name)) for name, score in person_best.items() if score >= threshold]
    matching.sort(key=lambda x: x[0], reverse=True)
    return [person for score, person in matching if person]


def _natural_list(names):
    """Return a human-readable list: 'Aaron, Rubi, and Ezza'."""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _first_name(person):
    return person.get("name", "Unknown").split()[0]


def _get_current_experience(person):
    experience = person.get("experience", [])
    return experience[0] if experience else None


def _first_sentence(text):
    if not text:
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return sentences[0] if sentences else text


def _group_people_by_value(people, value_getter):
    grouped = {}
    for person in people:
        value = value_getter(person)
        if not value:
            continue
        grouped.setdefault(value, []).append(_first_name(person))
    return grouped


def _format_grouped_topic_response(grouped, missing_message, multi_header, single_formatter=None):
    if not grouped:
        return missing_message

    if len(grouped) == 1 and single_formatter:
        value, names = list(grouped.items())[0]
        return single_formatter(names, value)

    parts = [f"{_natural_list(names)}: {value}" for value, names in grouped.items()]
    return f"{multi_header}{'; '.join(parts)}."


def _format_multi_response(people, topic):
    if not people:
        return "I couldn't find any team members matching that query."

    first_names = [_first_name(p) for p in people]

    # ---- Education ----
    if topic == "education":
        # Find a school attended by every person in the result set
        school_sets = [
            {e.get("school", "") for e in p.get("education", [])}
            for p in people
        ]
        common_schools = set.intersection(*school_sets) if school_sets else set()
        # Pick the most-mentioned common school as the primary
        primary_shared = None
        if common_schools:
            counts = {}
            for p in people:
                for e in p.get("education", []):
                    s = e.get("school", "")
                    if s in common_schools:
                        counts[s] = counts.get(s, 0) + 1
            primary_shared = max(counts, key=counts.get)

        if primary_shared and len(people) >= 2:
            intro = f"All {len(people)} team members — {_natural_list(first_names)} — attend {primary_shared}."
        else:
            # Group by school
            school_map = {}
            for p in people:
                first = _first_name(p)
                for e in p.get("education", []):
                    school = e.get("school", "N/A")
                    school_map.setdefault(school, []).append(first)
            parts = [f"{_natural_list(names)} at {school}" for school, names in school_map.items()]
            intro = "The team studies across a few schools: " + "; ".join(parts) + "."

        # Add detail for anyone with additional schools beyond the shared one
        extras = []
        for p in people:
            first = _first_name(p)
            edus = p.get("education", [])
            extra_schools = [e.get("school") for e in edus if e.get("school") != primary_shared]
            if extra_schools:
                extras.append(f"{first} also studied at {_natural_list(extra_schools)}")
        if extras:
            return intro + " " + ". ".join(extras) + "."
        return intro

    # ---- Skills ----
    if topic == "skills":
        # Find skills common to all
        skill_sets = [set(p.get("skills", [])) for p in people if p.get("skills")]
        if skill_sets:
            common = set.intersection(*skill_sets)
        else:
            common = set()

        parts = []
        for p in people:
            first = _first_name(p)
            skills = p.get("skills", [])
            if skills:
                unique = [s for s in skills if s not in common]
                if unique:
                    parts.append(f"{first} brings {', '.join(unique[:3])}")
                else:
                    parts.append(f"{first} covers the shared stack")

        if common:
            shared_str = ", ".join(sorted(common)[:5])
            intro = f"The whole team shares skills in {shared_str}."
        else:
            intro = f"Here's what {_natural_list(first_names)} each bring to the table:"

        if parts:
            return intro + " " + ". ".join(parts) + "."
        return intro

    # ---- Location ----
    if topic == "location":
        locs = {}
        for p in people:
            loc = p.get("location")
            if loc:
                locs.setdefault(loc, []).append(_first_name(p))
        if len(locs) == 1:
            loc, names = list(locs.items())[0]
            return f"The whole team is based in {loc}."
        parts = [f"{_natural_list(names)} in {loc}" for loc, names in locs.items()]
        return f"The team is spread across a few areas: {'; '.join(parts)}."

    # ---- Experience / Employer ----
    if topic in ("experience", "employer"):
        parts = []
        for p in people:
            first = _first_name(p)
            current = _get_current_experience(p)
            if current:
                parts.append(f"{first} is currently a {current.get('title', 'professional')} at {current.get('company', 'a company')}")
        if not parts:
            return f"I don't have current employer information for the team right now."
        return "Here's where the team currently works: " + "; ".join(parts) + "."

    # ---- Availability ----
    if topic == "availability":
        grouped = _group_people_by_value(people, lambda p: p.get("availability"))
        return _format_grouped_topic_response(
            grouped,
            "I don't have availability information for the team right now.",
            "Team availability: ",
            single_formatter=lambda names, value: f"{_natural_list(names)} {'are' if len(names) > 1 else 'is'} {value}."
        )

    # ---- Authorization ----
    if topic == "authorization":
        grouped = _group_people_by_value(people, lambda p: p.get("work_authorization"))
        return _format_grouped_topic_response(
            grouped,
            "I don't have work authorization details for the team right now.",
            "Work authorization across the team: ",
            single_formatter=lambda names, value: f"{_natural_list(names)} {'are' if len(names) > 1 else 'is'} {value}."
        )

    # ---- Remote preference ----
    if topic == "remote_preference":
        grouped = _group_people_by_value(people, lambda p: p.get("remote_preference"))
        return _format_grouped_topic_response(
            grouped,
            "I don't have remote/onsite preferences for the team right now.",
            "Work setting preferences across the team: "
        )

    # ---- Relocation ----
    if topic == "relocation":
        grouped = _group_people_by_value(people, lambda p: p.get("relocation"))
        return _format_grouped_topic_response(
            grouped,
            "I don't have relocation preferences for the team right now.",
            "Relocation preferences across the team: ",
            single_formatter=lambda names, value: f"{_natural_list(names)} {'are' if len(names) > 1 else 'is'} open to {value}."
        )

    # ---- Preferred roles ----
    if topic == "preferred_roles":
        parts = []
        for p in people:
            first = _first_name(p)
            roles = p.get("preferred_roles", [])
            if roles:
                parts.append(f"{first} is targeting {', '.join(roles[:2])}")
        if not parts:
            return "I don't have preferred role information for the team right now."
        return ". ".join(parts) + "."

    # ---- About / overview ----
    if topic == "about":
        parts = []
        for p in people:
            name = p.get("name", "Unknown")
            headline = p.get("headline", "")
            about = p.get("about", "")
            location = p.get("location", "")
            skills = p.get("skills", [])[:4]
            summary = _first_sentence(about)
            lines = [f"{name}"]
            if headline:
                lines.append(f"Role: {headline}")
            if location:
                lines.append(f"Location: {location}")
            if skills:
                lines.append(f"Skills: {', '.join(skills)}")
            if summary:
                lines.append(f"{summary}")
            parts.append("\n".join(lines))
        if not parts:
            return "I don't have overview information for the team right now."
        intro = f"Here's a snapshot of our {len(people)} team members:"
        return intro + "\n\n" + "\n\n".join(parts)

    # ---- Fallback ----
    parts = []
    for p in people:
        first = _first_name(p)
        about = p.get("about") or p.get("headline", "")
        if about:
            parts.append(f"{first}: {about[:100]}")
    if not parts:
        return f"I don't have {topic.replace('_', ' ')} information available for the team right now."
    return "Here's a quick overview of the team: " + " | ".join(parts) + "."


def _retrieve_context(query, matched_person=None, matched_topic=None, top_k=4):
    query_vec = _tfidf_vector(query, IDF)
    if not query_vec:
        return []
    ranked = []

    for chunk, chunk_vec in zip(CHUNKS, CHUNK_VECTORS):
        score = _cosine_similarity(query_vec, chunk_vec)

        if matched_person and chunk.get("person") == matched_person.get("name"):
            score += 0.2

        if matched_topic and chunk.get("topic") == matched_topic:
            score += 0.15

        if score > 0:
            ranked.append((score, chunk))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[:top_k]


_LINKEDIN_ONLY_TOPICS = {"compensation"}


def _grounded_fallback(user_query, matched_person, matched_topic, context_chunks):
    if matched_person:
        if matched_topic in _LINKEDIN_ONLY_TOPICS:
            name = matched_person.get("name", "This person")
            linkedin = matched_person.get("contact", {}).get("linkedin", "")
            return (
                f"Compensation details aren't listed in {name}'s profile. "
                + (f"Reach out via LinkedIn: {linkedin}." if linkedin else "Reach out to them directly.")
            )
        if matched_topic is None:
            return _format_full_profile(matched_person)
        return _format_response(matched_person, matched_topic)

    if context_chunks:
        top_score, top_chunk = context_chunks[0]
        if top_score >= MIN_RETRIEVAL_CONFIDENCE and matched_topic:
            return f"Based on the knowledge base: {top_chunk['text']}"

    if matched_topic and not matched_person:
        if matched_topic in _LINKEDIN_ONLY_TOPICS:
            topic_label = matched_topic.replace("_", " ")
            return (
                f"Compensation details aren't included in our LinkedIn profile data. "
                f"Reach out directly to each team member via their LinkedIn for that discussion."
            )
        return "Bob here. Who are you asking about? Try Aaron, Rubi, Ezza, Rasheed, or Albion."

    return "Bob here. I don't have info on that yet. Try asking about Aaron, Rubi, Ezza, Rasheed, or Albion."


def _format_response(person, topic):
    name = person.get("name", "This person")
    first_name = _first_name(person)

    def _simple_field_response(value, success_template, missing_template):
        if value:
            return success_template.format(first_name=first_name, name=name, value=value)
        return missing_template.format(first_name=first_name, name=name)

    if topic == "skills":
        skills = person.get("skills", [])
        if not skills:
            return f"I don't have skills info for {name} right now."
        skills_str = ", ".join(skills[:-1]) + f", and {skills[-1]}" if len(skills) > 1 else skills[0]
        return f"{first_name}'s skills include {skills_str}."

    if topic == "education":
        education = person.get("education", [])
        if not education:
            return f"I don't have education details for {name} at the moment."
        if len(education) == 1:
            e = education[0]
            return f"{first_name} is pursuing a {e.get('degree', 'degree')} in {e.get('field', 'a field')} at {e.get('school', 'a university')}."
        else:
            lines = [f"{e.get('degree', 'Degree')} in {e.get('field', 'N/A')} from {e.get('school', 'N/A')}" for e in education]
            return f"{first_name}'s education: {'; '.join(lines)}."

    if topic == "experience":
        current = _get_current_experience(person)
        if not current:
            return f"{first_name} hasn't listed work experience yet."
        experience = person.get("experience", [])
        base = f"{first_name} works as a {current.get('title', 'role')} at {current.get('company', 'a company')}."
        if len(experience) > 1:
            other = ", ".join(e.get('title', 'a role') for e in experience[1:])
            base += f" Previously: {other}."
        return base

    if topic == "employer":
        current = _get_current_experience(person)
        if not current:
            return f"I don't have employer information for {name} right now."
        return f"{first_name} works at {current.get('company', 'a company')} as a {current.get('title', 'team member')}."

    if topic == "location":
        return _simple_field_response(
            person.get("location"),
            "{first_name} is based in {value}.",
            "I don't have location info for {name}."
        )

    if topic == "headline":
        return _simple_field_response(
            person.get("headline"),
            "{first_name} is a {value}.",
            "No headline listed for {name}."
        )

    if topic == "availability":
        return _simple_field_response(
            person.get("availability"),
            "{first_name}: {value}.",
            "No availability info for {name} yet."
        )

    if topic == "preferred_roles":
        preferred_roles = person.get("preferred_roles", [])
        if preferred_roles:
            roles_str = ", ".join(preferred_roles[:-1]) + f", and {preferred_roles[-1]}" if len(preferred_roles) > 1 else preferred_roles[0]
            return f"{first_name} is targeting: {roles_str}."
        return f"No preferred role info for {name} yet."

    if topic == "remote_preference":
        return _simple_field_response(
            person.get("remote_preference"),
            "{first_name}: {value}.",
            "No remote/onsite preference listed for {name}."
        )

    if topic == "relocation":
        return _simple_field_response(
            person.get("relocation"),
            "{first_name}: {value}.",
            "No relocation info for {name} yet."
        )

    if topic == "contact":
        contact = person.get("contact", {})
        linkedin = contact.get("linkedin")
        return f"{first_name}'s LinkedIn: {linkedin}." if linkedin else f"No LinkedIn profile listed for {name} yet."

    if topic == "compensation":
        return (
            f"Compensation details aren't listed in {name}'s profile. "
            f"Reach out via LinkedIn to discuss: {person.get('contact', {}).get('linkedin', 'see their contact info')}."
        )

    if topic == "authorization":
        return _simple_field_response(
            person.get("work_authorization"),
            "{first_name}: {value}.",
            "No work authorization info for {name} yet."
        )

    about = person.get("about")
    if about or person.get("headline"):
        return _format_full_profile(person)
    return f"I don't have a summary for {name} yet."


def _format_full_profile(person):
    """Return a concise profile summary for a single person."""
    name       = person.get("name", "This person")
    headline   = person.get("headline", "")
    location   = person.get("location", "")
    about      = person.get("about", "")
    skills     = person.get("skills", [])
    roles      = person.get("preferred_roles", [])

    lines = []

    # Header line
    header = name
    if headline:
        header += f"  ·  {headline}"
    if location:
        header += f"  ·  {location}"
    lines.append(header)
    lines.append("")

    # About paragraph
    if about:
        lines.append(about)
        lines.append("")

    # Skills (single line)
    if skills:
        lines.append("Skills: " + " · ".join(skills))
        lines.append("")

    # Preferred roles (single line)
    if roles:
        lines.append("Looking for: " + ", ".join(roles))

    return "\n".join(lines).strip()


TEAM_NAMES = [p.get("name", "") for p in KNOWLEDGE_BASE]
TEAM_FIRST_NAMES = [n.split()[0] for n in TEAM_NAMES]

TEAM_WHO_PATTERNS = [
    r"\bwho(?:'s| is| are)? (?:in|on|part of)? ?(?:the )?team\b",
    r"\bwho(?:'s| is| are)? (?:the )?(?:team )?members?\b",
    r"\blist (?:all )?(?:the )?(?:team )?members?\b",
    r"\bteam (?:members?|roster|lineup)\b",
    r"\b(?:tell me about|about|describe) (?:all|every|each) (?:(?:team |group )?members?|everyone|everybody)\b",
    r"\b(?:all|every|each) (?:(?:team |group )?members?|everyone|everybody)\b",
]

TEAM_WHO_REGEX = [re.compile(pattern) for pattern in TEAM_WHO_PATTERNS]

TEAM_WHY_PATTERNS = [
    r"\bwhy (?:are|is) (?:this|the|they|you) (?:in )?(?:a )?team\b",
    r"\bwhat (?:is|'?s) (?:this )?(?:team|group|project)\b",
    r"\bwhat (?:class|course|subject)\b",
    r"\bwhy (?:did|do) (?:you|they) (?:work|team) together\b",
    r"\bgroup project\b",
    r"\bcs\s*335\b",
]

TEAM_WHY_REGEX = [re.compile(pattern) for pattern in TEAM_WHY_PATTERNS]

BOT_INTENT_PATTERNS = [
    r"\bwhat (?:are|can) you (?:do|help|answer|tell)\b",
    r"\bwhat (?:is|'?s) (?:this )?(?:chatbot|bot|assistant|tool|app|application)\b",
    r"\bhow (?:does|do) (?:this|you|the) (?:chatbot|bot|work|assistant)\b",
    r"\bwhat (?:do|can) you know\b",
    r"\bwhat (?:kind|type) of (?:questions|info|information)\b",
    r"\bwho (?:are|is) (?:you|bob)\b",
    r"\bwhat (?:are|is) (?:your|bob'?s?) (?:purpose|goal|function|job|role|intent)\b",
    r"\bwhat (?:is|'?s) (?:the )?purpose of (?:this|the) (?:chatbot|bot|assistant|tool)\b",
    r"\bhow (?:can|do|should) i (?:use|ask|talk to) (?:you|this|bob)\b",
    r"\bwhat (?:questions|topics) (?:can|should) i ask\b",
    r"\bintroduce yourself\b",
    r"\bwhat do you do\b",
]

BOT_INTENT_REGEX = [re.compile(p) for p in BOT_INTENT_PATTERNS]


def _handle_bot_intent(raw_input):
    lowered = raw_input.lower()
    if not any(pattern.search(lowered) for pattern in BOT_INTENT_REGEX):
        return None
    return (
        "I'm Bob, a knowledge assistant built for Group 2's CS 335 final project at NEIU. "
        "I can answer recruiter and general questions about our five team members — "
        "Aaron, Rubi, Ezza, Rasheed, and Albion — using information from their LinkedIn profiles.\n\n"
        "Here's what you can ask me:\n"
        "  • Background & summary"
        "  • Skills"
        "  • Education"
        "  • Experience"
        "  • Availability"
        "  • Remote/hybrid preference"
        "  • Relocation"
        "  • Work authorization"
        "  • Preferred roles"
        "  • Contact"
        "  • Team info"
    )


def _handle_team_meta(raw_input):
    lowered = raw_input.lower()

    # Who is in the team?
    if any(pattern.search(lowered) for pattern in TEAM_WHO_REGEX):
        return "Our team has 5 members: " + ", ".join(TEAM_NAMES) + "."

    # Why are they a team / what is this project?
    if any(pattern.search(lowered) for pattern in TEAM_WHY_REGEX):
        names_str = _natural_list(TEAM_FIRST_NAMES)
        return (
            f"We're Group 2 — a team of five CS students at Northeastern Illinois University. "
            f"{names_str} came together for CS 335 (Artificial Intelligence) to build this chatbot as our final project. "
            f"All information is sourced from our LinkedIn profiles. Ask me about any of us and I'll tell you more."
        )

    return None


# ---------------- RETRIEVAL FUNCTION ----------------
def get_best_response(user_input, conversation_context=None):
    clean = _expand_abbreviations(user_input.strip())

    # Edge case: empty input
    if clean == "":
        return "Bob here. I'm powered by our team's LinkedIn profiles. Ask me about Aaron, Rubi, Ezza, Rasheed, or Albion.", conversation_context or {}

    # Check for bot intent questions first (what can you do, who are you, etc.)
    bot_intent = _handle_bot_intent(clean)
    if bot_intent:
        return bot_intent, conversation_context or {}

    # Check for team meta questions first (who is in the team, why are you a team, etc.)
    team_meta = _handle_team_meta(clean)
    if team_meta:
        return team_meta, conversation_context or {}

    # Check for generic/greeting inputs
    generic_response = _handle_generic_input(clean)
    if generic_response:
        return generic_response, conversation_context or {}

    conversation_context = conversation_context or {}
    raw_tokens = set(re.findall(r"\b\w+\b", clean.lower()))
    tokens = set(_tokenize(clean))
    if not tokens:
        tokens = raw_tokens

    matched_person = _find_person(tokens)
    matched_topic = _find_topic(tokens)

    # "they/them/their" can mean the whole team (no prior context) or follow-up on last person
    PLURAL_PRONOUNS = {"they", "them"}
    uses_plural_pronoun = bool(raw_tokens & PLURAL_PRONOUNS)
    has_prev_person = bool(conversation_context.get("last_person"))

    # Don't apply previous-person context if:
    # 1. It's a broad team query, OR
    # 2. "they/them" is used but there's no previous person (implies team-wide question)
    if (not matched_person
            and not _is_broad_query(raw_tokens)
            and not (uses_plural_pronoun and not has_prev_person)
            and (raw_tokens & PRONOUN_HINTS or (matched_topic and raw_tokens & FOLLOW_UP_CONTEXT_HINTS))):
        matched_person = PERSON_BY_NAME.get(conversation_context.get("last_person"))

    # If "they/them" with no prior context, treat as broad query
    if not matched_person and uses_plural_pronoun and not has_prev_person:
        raw_tokens = raw_tokens | {"all"}  # inject broad trigger

    if not matched_topic and (raw_tokens & PRONOUN_HINTS or raw_tokens & FOLLOW_UP_CONTEXT_HINTS):
        matched_topic = conversation_context.get("last_topic")

    # Multi-person broad query: "who attends NEIU", "who works at UPS", "who knows Python", etc.
    if not matched_person and _is_broad_query(raw_tokens):
        # Strip broad/filler words and JSON key names to get meaningful content tokens
        content_tokens = raw_tokens - BROAD_QUERY_WORDS - STOP_WORDS - BROAD_FILLER_WORDS - JSON_KEY_WORDS

        # Also extract special tech names (C++, C#, .NET) that the tokenizer strips
        tech_names = [t.lower() for t in _extract_tech_names(clean)]

        # Resolve topic alias (employer -> experience, headline -> about)
        inferred_topic = TOPIC_CHUNK_ALIAS.get(matched_topic, matched_topic)

        direct_matches = []

        # Step 1: Direct literal scan of JSON values when there are specific content tokens
        if content_tokens or tech_names:
            all_terms = [re.escape(t) for t in content_tokens] + [re.escape(t) for t in tech_names]
            term_group = "|".join(all_terms)
            pattern = re.compile(
                r'"(?:' + term_group + r')"'                         # exact value: "Python"
                r'|"[^"]*\b(?:' + term_group + r')\b[^"]*"',        # within value: "Chicago, IL"
                re.IGNORECASE
            )
            for person in KNOWLEDGE_BASE:
                if pattern.search(PERSON_JSON_CACHE[person.get("name", "")]):
                    direct_matches.append(person)

        if direct_matches:
            # Infer topic from top-scoring chunk when not already known
            if not inferred_topic:
                query_vec = _tfidf_vector(clean, IDF)
                ranked = sorted(
                    ((_cosine_similarity(query_vec, cv), ch) for ch, cv in zip(CHUNKS, CHUNK_VECTORS)),
                    key=lambda x: x[0], reverse=True
                )
                inferred_topic = ranked[0][1].get("topic") if ranked else "skills"

            if len(direct_matches) > 1:
                new_context = {"last_person": conversation_context.get("last_person"), "last_topic": inferred_topic}
                return _format_multi_response(direct_matches, inferred_topic), new_context
            else:
                matched_person = direct_matches[0]
                matched_topic = inferred_topic

        else:
            # Step 2: No direct value match — use TF-IDF retrieval, topic-scoped when possible
            # When a topic keyword was matched (e.g. "experience", "education"), return ALL people
            # for that topic since there are no filtering content tokens
            if inferred_topic:
                # Filter to only people who actually have data for this topic
                all_people = [p for p in KNOWLEDGE_BASE if _has_topic_data(p, inferred_topic)]
                if not all_people:
                    topic_label = inferred_topic.replace("_", " ")
                    return (
                        f"That information ({topic_label}) isn't included in our LinkedIn profile data. "
                        f"Try asking about skills, education, experience, location, or preferred roles."
                    ), conversation_context
                if len(all_people) > 1:
                    new_context = {"last_person": conversation_context.get("last_person"), "last_topic": inferred_topic}
                    return _format_multi_response(all_people, inferred_topic), new_context
                elif len(all_people) == 1:
                    matched_person = all_people[0]
                    matched_topic = inferred_topic
            else:
                # Step 3: No topic and no direct match — relaxed TF-IDF inference
                query_vec = _tfidf_vector(clean, IDF)
                ranked = sorted(
                    ((_cosine_similarity(query_vec, cv), ch) for ch, cv in zip(CHUNKS, CHUNK_VECTORS)),
                    key=lambda x: x[0], reverse=True
                )
                if ranked and ranked[0][0] >= 0.12:
                    inferred_topic = ranked[0][1].get("topic")
                    all_people = _find_all_people_for_topic(clean, inferred_topic, threshold=0.08)
                    if len(all_people) > 1:
                        new_context = {"last_person": conversation_context.get("last_person"), "last_topic": inferred_topic}
                        return _format_multi_response(all_people, inferred_topic), new_context
                    elif len(all_people) == 1:
                        matched_person = all_people[0]
                        matched_topic = inferred_topic

    context_chunks = _retrieve_context(clean, matched_person=matched_person, matched_topic=matched_topic, top_k=4)
    response = _grounded_fallback(clean, matched_person, matched_topic, context_chunks)

    new_context = {
        "last_person": matched_person.get("name") if matched_person else conversation_context.get("last_person"),
        "last_topic": matched_topic if matched_topic else conversation_context.get("last_topic")
    }

    return response, new_context

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    conversation_context = session.get("conversation_context", {})
    bot_reply, updated_context = get_best_response(user_message, conversation_context)
    session["conversation_context"] = updated_context
    return jsonify({"response": bot_reply})

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    app.run(debug=True)
