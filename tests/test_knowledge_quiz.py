import random

import pytest

from core.knowledge import normalize_text
from core.quiz import TOPICS, build_quiz, generate_questions, load_bank, score_answer


@pytest.mark.parametrize("label, expected", [
    ("Búa", "bua"), ("0 Búa", "bua"), ("Cờ_lê", "co_le"), ("tua vit", "tua_vit"), ("Tô vít", "tua_vit"),
    ("Wrench", "co_le"), ("Giũa", "giua"), ("Dũa", "giua"), ("Cưa sắt", "cua_tay"), ("Kìm mỏ nhọn", "kim"),
    ("Hammer", "bua"), ("Mỏ lết", "mo_let"), ("Ê tô", "e_to"), ("Thước cặp", "thuoc_cap"),
])
def test_match_label(kb, label, expected):
    assert kb.match_label(label).id == expected


@pytest.mark.parametrize("label", ["Background", "Nền", "Khác", "none"])
def test_background_labels(kb, label):
    assert kb.is_background_label(label)
    assert kb.match_label(label) is None


def test_unknown_label(kb):
    assert kb.match_label("Con mèo") is None
    assert not kb.is_background_label("Con mèo")


def test_normalize_text():
    assert normalize_text("  Đục_SẮT!! ") == "duc sat"


def test_knowledge_base_complete(kb):
    assert len(kb) == 15
    for tool in kb.all():
        assert tool.summary.startswith("dùng để"), tool.id
        assert tool.uses and tool.safety and tool.structure and tool.how_to_use, tool.id
        assert kb.group(tool.group)


def test_search_ignores_accents(kb):
    ids = {t.id for t in kb.search("cat day")}
    assert "kim" in ids
    assert {t.id for t in kb.search("")} == set(kb.ids())


def test_bank_is_valid():
    bank = load_bank()
    assert len(bank) >= 30
    assert len({q.id for q in bank}) == len(bank)
    for q in bank:
        assert 0 <= q.answer < len(q.options)
        assert len(set(q.options)) == len(q.options), q.id


def test_generated_questions_have_unique_options(kb):
    for q in generate_questions(kb, random.Random(1)):
        assert len(set(q.options)) == len(q.options), q.id
        assert q.correct_option


@pytest.mark.parametrize("topic", list(TOPICS))
def test_build_quiz(kb, topic):
    questions = build_quiz(kb, load_bank(), topic, 10, seed=3)
    assert len(questions) == 10
    assert len({q.id for q in questions}) == 10
    if topic != "all":
        assert all(q.topic == topic for q in questions)


def test_shuffle_keeps_correct_answer(kb):
    bank = load_bank()
    original = {q.id: q.correct_option for q in bank}
    for q in build_quiz(kb, bank, "all", 15, seed=11):
        if q.source == "bank":
            assert q.correct_option == original[q.id]


def test_score_answer_streak_bonus(kb):
    q = load_bank()[0]
    streak, xps = 0, []
    for _ in range(4):
        outcome = score_answer(q, q.answer, streak)
        streak = outcome.streak
        xps.append(outcome.xp)
    assert xps == [10, 10, 15, 15]
    wrong = score_answer(q, (q.answer + 1) % len(q.options), streak)
    assert (wrong.correct, wrong.xp, wrong.streak) == (False, 0, 0)
