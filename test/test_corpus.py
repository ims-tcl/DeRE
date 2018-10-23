import pytest

from dere.corpus import Instance, Corpus, Span, Slot, Frame


class MockFrameType:
    def __init__(self, iterable):
        self.slot_types = iterable


class MockInstance:
    def __init__(self, *, text=None, spans=None, frames=None):
        self.text = text
        self.spans = spans
        self.frames = frames


class MockSlotType:
    def __init__(self, name):
        self.name = name


def test_frame_slot_lookup():
    f = Frame(
        frame_type=MockFrameType(list(map(MockSlotType, ["foo", "bar", "baz"]))),
        instance=None,
        source="doesn't matter",
    )
    assert len(f.slots) == 3
    assert isinstance(f.slot_lookup("foo"), Slot)
    assert f.slot_lookup("bat") is None


def test_frame_remove():
    frames = [
        Frame(frame_type=MockFrameType([]), instance=None, source=""),
        Frame(frame_type=MockFrameType([]), instance=None, source=""),
    ]
    f = Frame(
        frame_type=MockFrameType([]), instance=MockInstance(frames=frames),
        source="doesn't matter",
    )
    f.instance.frames.append(f)
    f.remove()
    assert len(f.instance.frames) == 2

    with pytest.raises(ValueError):
        f.remove()


def test_span_bad_values():
    with pytest.raises(ValueError):
        # left has to be <= right
        Span("spantype", 86, 37, None, None, None)


def test_span_text():
    c = Corpus()
    i = c.new_instance("It should capture ->this<-, nothing else", "docid")
    s = i.new_span("spantype", 20, 24)
    assert s.text == "this"


def test_span_remove():
    s = Span("spantype", 1, 2, MockInstance(spans=[1, 2, 3]), None, None)
    s.instance.spans.append(s)
    assert len(s.instance.spans) == 4
    s.remove()
    assert len(s.instance.spans) == 3
    with pytest.raises(ValueError):
        s.remove()


def test_slot():
    s = Slot("slottype", "I'm a frame")
    assert not s.fillers

    for i in range(7):
        s.add(f"filler_{i}")
    assert len(s.fillers) == 7


def test_instance():
    c = Corpus()
    i = Instance("some text", "docid", c)
    assert not i.spans
    assert not i.frames

    s = i.new_span("some type", 42, 56)
    assert isinstance(s, Span)
    assert i.spans
    assert not i.frames

    f = i.new_frame(MockFrameType([]))
    assert isinstance(f, Frame)
    assert i.frames

    f = i.new_frame(MockFrameType(["typea", "typeb"]))
    assert len(f.slots) == 2


def test_corpus():
    c = Corpus()
    assert len(c.instances) == 0
    i = c.new_instance("string", "docid")
    assert isinstance(i, Instance)
    assert len(c.instances) == 1
    c.new_instance("foo", "docid")
    c.new_instance("foo", "docid")
    assert len(c.instances) == 3
