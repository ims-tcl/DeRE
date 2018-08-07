import pytest
import networkx as nx
from dere.models._baseline.slot_classifier import SlotClassifier


class MockTaskSpec:
    def __init__(self):
        self.frame_types = []


class MockToken:
    @classmethod
    def words(self, words, idx=0):
        res = []
        for i, word in enumerate(words.split()):
            res.append(MockToken(word, idx + i))
        return res

    def __init__(self, word, idx):
        self.word = word
        self.idx = idx


empty_graph = nx.Graph()
for i in range(20):
    empty_graph.add_node(i)

fully_connected_graph = nx.Graph()
for i in range(20):
    fully_connected_graph.add_node(i)
    for j in range(i):
        fully_connected_graph.add_edge(i, j)

interesting_graph = nx.Graph()
for i in range(8):
    interesting_graph.add_node(i)

for edge in [(0, 2), (1, 2), (2, 3), (3, 4), (4, 7), (5, 7), (6, 7)]:
    interesting_graph.add_edge(*edge)


@pytest.mark.parametrize(
    "graph,tokens1,tokens2,result",
    [
        (
            empty_graph,
            MockToken.words("foo bar bat"),
            MockToken.words("bla blubb boo", idx=5),
            None,
        ),
        (
            empty_graph,
            MockToken.words("foo bar bat"),
            MockToken.words("bla blubb boo", idx=0),
            [0],
        ),
        (
            fully_connected_graph,
            MockToken.words("foo bar bat"),
            MockToken.words("bla blubb boo", idx=5),
            [0, 5],
        ),
        (
            interesting_graph,
            MockToken.words("small cat", idx=1),
            MockToken.words("the big", idx=5),
            5,
        ),
    ],
)
def test_get_shortest_path(graph, tokens1, tokens2, result):
    sc = SlotClassifier(MockTaskSpec())
    if isinstance(result, int):
        assert len(sc.get_shortest_path(graph, tokens1, tokens2)) == result
    else:
        assert sc.get_shortest_path(graph, tokens1, tokens2) == result


@pytest.mark.parametrize(
    "graph,tokens1,tokens2,idx2word,result",
    [
        (
            empty_graph,
            MockToken.words("foo bar bat"),
            MockToken.words("bla blubb boo", idx=5),
            {0: "foo", 1: "bar", 2: "bat", 3: "bla", 4: "blubb", 5: "boo"},
            [],
        ),
        (
            fully_connected_graph,
            MockToken.words("foo bar bat"),
            MockToken.words("bla blubb boo", idx=3),
            {0: "foo", 1: "bar", 2: "bat", 3: "bla", 4: "blubb", 5: "boo"},
            ["foo", "bla"],
        ),
        # (interesting_graph, MockToken.words("small cat", idx=1), MockToken.words("the big", idx=5), 5),
    ],
)
def test_edge_words(graph, tokens1, tokens2, idx2word, result):
    sc = SlotClassifier(MockTaskSpec())
    assert sc.edge_words(graph, tokens1, tokens2, idx2word) == result


@pytest.mark.parametrize(
    "graph,tokens1,tokens2,idx2word,edge2dep,result",
    [
        (
            empty_graph,
            MockToken.words("foo bar bat"),
            MockToken.words("bla blubb boo", idx=5),
            {0: "foo", 1: "bar", 2: "bat", 3: "bla", 4: "blubb", 5: "boo"},
            {(0, 1): "N", (1, 2): "VP", (2, 3): "Det", (3, 4): "PP", (4, 5): "ADJ"},
            "",
        ),
        (
            fully_connected_graph,
            MockToken.words("foo bar bat"),
            MockToken.words("bla blubb boo", idx=3),
            {0: "foo", 1: "bar", 2: "bat", 3: "bla", 4: "blubb", 5: "boo"},
            {
                (0, 1): "N",
                (0, 3): "N",
                (0, 2): "N",
                (1, 2): "VP",
                (1, 3): "VP",
                (2, 3): "Det",
                (2, 1): "Det",
                (3, 4): "PP",
                (4, 5): "ADJ",
            },
            "foo N bla",
        ),
        # (interesting_graph, MockToken.words("small cat", idx=1), MockToken.words("the big", idx=5), 5),
    ],
)
def test_edge_words_deps(graph, tokens1, tokens2, idx2word, edge2dep, result):
    sc = SlotClassifier(MockTaskSpec())
    assert sc.edge_words_deps(graph, tokens1, tokens2, idx2word, edge2dep) == result


