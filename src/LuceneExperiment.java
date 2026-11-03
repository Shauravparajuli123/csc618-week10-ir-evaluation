import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.MultiTerms;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.FuzzyQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.PrefixQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TermRangeQuery;
import org.apache.lucene.search.WildcardQuery;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;
import org.apache.lucene.search.similarities.LMJelinekMercerSimilarity;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;

/**
 * Indexes Cranfield in Lucene, then produces all ranked result files.
 *
 * <p>Self Study 1: eight query classes per topic (top 100). Part C: retrieval models (top 1000).
 *
 * <p>Usage: java LuceneExperiment &lt;dataDir&gt; &lt;resultsDir&gt;
 */
public class LuceneExperiment {
  private static final String FIELD = "contents";
  private static final String LENGTH_FIELD = "len";
  private static final Analyzer ANALYZER = new EnglishAnalyzer();

  /** One analysed token and its position (positions keep gaps left by removed stop words). */
  record Token(String term, int position) {}

  /** Document-length summary used by the numeric range query. */
  record LengthStats(double mean, int q1, int q3) {}

  public static void main(String[] args) throws Exception {
    Path dataDir = Paths.get(args[0]);
    Path resultsDir = Paths.get(args[1]);
    Files.createDirectories(resultsDir);

    Directory directory = new ByteBuffersDirectory();
    LengthStats lengths = buildIndex(dataDir, resultsDir, directory);
    List<String[]> queries = readQueries(dataDir);

    try (IndexReader reader = DirectoryReader.open(directory)) {
      writeIndexStats(reader, lengths, resultsDir);
      IndexSearcher bm25Searcher = new IndexSearcher(reader);
      bm25Searcher.setSimilarity(new BM25Similarity());

      runQueryTypeExperiment(reader, bm25Searcher, queries, lengths, resultsDir);
      runDemoQueries(reader, bm25Searcher, lengths, resultsDir);
      runRetrievalModels(reader, queries, resultsDir);
    }
  }

  // ---------------------------------------------------------------- Indexing

  /** Indexes every document and also saves its analysed terms for the KLD/JSD calculations. */
  private static LengthStats buildIndex(Path dataDir, Path resultsDir, Directory directory)
      throws IOException {
    List<Integer> lengths = new ArrayList<>();
    IndexWriterConfig config = new IndexWriterConfig(ANALYZER).setSimilarity(new BM25Similarity());

    try (IndexWriter writer = new IndexWriter(directory, config);
        PrintWriter analysed = new PrintWriter(resultsDir.resolve("docs_analyzed.tsv").toFile())) {
      for (String line : Files.readAllLines(dataDir.resolve("docs.tsv"))) {
        String[] parts = line.split("\t", -1);
        String id = parts[0];
        String title = parts[1];
        String text = parts[2];
        // The <text> field already begins with the title; two documents have no text at all.
        String body = text.isEmpty() ? title : text;

        List<Token> tokens = analyze(body);
        lengths.add(tokens.size());

        Document document = new Document();
        document.add(new StringField("id", id, Field.Store.YES));
        document.add(new TextField(FIELD, body, Field.Store.NO));
        document.add(new IntPoint(LENGTH_FIELD, tokens.size()));
        writer.addDocument(document);

        StringBuilder row = new StringBuilder(id).append('\t');
        for (Token token : tokens) {
          row.append(token.term()).append(' ');
        }
        analysed.println(row.toString().trim());
      }
    }

    Collections.sort(lengths);
    int q1 = lengths.get(lengths.size() / 4);
    int q3 = lengths.get(3 * lengths.size() / 4);
    double mean = lengths.stream().mapToInt(Integer::intValue).average().orElse(0);
    return new LengthStats(mean, q1, q3);
  }

  private static void writeIndexStats(IndexReader reader, LengthStats lengths, Path resultsDir)
      throws IOException {
    try (PrintWriter out = new PrintWriter(resultsDir.resolve("index_stats.txt").toFile())) {
      out.printf(
          Locale.ROOT,
          "lucene=%s docs=%d vocab=%d tokens=%d meanLen=%.2f q1=%d q3=%d%n",
          org.apache.lucene.util.Version.LATEST,
          reader.numDocs(),
          MultiTerms.getTerms(reader, FIELD).size(),
          reader.getSumTotalTermFreq(FIELD),
          lengths.mean(),
          lengths.q1(),
          lengths.q3());
    }
  }

  // ---------------------------------------------------------------- Helpers

  /** Applies the same analysis to queries as to documents (stop words, Porter stemming). */
  private static List<Token> analyze(String text) throws IOException {
    List<Token> tokens = new ArrayList<>();
    try (TokenStream stream = ANALYZER.tokenStream(FIELD, text)) {
      CharTermAttribute term = stream.addAttribute(CharTermAttribute.class);
      PositionIncrementAttribute increment = stream.addAttribute(PositionIncrementAttribute.class);
      stream.reset();
      int position = -1;
      while (stream.incrementToken()) {
        position += increment.getPositionIncrement();
        tokens.add(new Token(term.toString(), position));
      }
      stream.end();
    }
    return tokens;
  }

  private static List<String[]> readQueries(Path dataDir) throws IOException {
    List<String[]> queries = new ArrayList<>();
    for (String line : Files.readAllLines(dataDir.resolve("queries.tsv"))) {
      queries.add(line.split("\t", -1)); // sequential id, original number, text
    }
    return queries;
  }

  private static Term term(String text) {
    return new Term(FIELD, text);
  }

  /** OR of all query terms: a document needs at least one of them to match. */
  private static Query orQuery(List<Token> tokens) {
    BooleanQuery.Builder builder = new BooleanQuery.Builder();
    for (Token token : tokens) {
      builder.add(new TermQuery(term(token.term())), BooleanClause.Occur.SHOULD);
    }
    return builder.build();
  }

  /** Inverse document frequency, or -1 when the term is not in the index. */
  private static double idf(IndexReader reader, String text) throws IOException {
    int documentFrequency = reader.docFreq(term(text));
    return documentFrequency == 0 ? -1 : Math.log((reader.numDocs() + 1.0) / documentFrequency);
  }

  /** Runs a query and writes results in TREC format: qid Q0 docid rank score tag. */
  private static void writeRun(
      IndexSearcher searcher, Query query, int queryId, String tag, int depth, PrintWriter out)
      throws IOException {
    StoredFields fields = searcher.getIndexReader().storedFields();
    ScoreDoc[] hits = searcher.search(query, depth).scoreDocs;
    for (int i = 0; i < hits.length; i++) {
      String docId = fields.document(hits[i].doc).get("id");
      out.printf(Locale.ROOT, "%d Q0 %s %d %.6f %s%n", queryId, docId, i + 1, hits[i].score, tag);
    }
  }

  // ---------------------------------------------------------------- Self Study 1

  private static String rarestTerm(IndexReader reader, List<Token> tokens) throws IOException {
    String best = null;
    double bestIdf = -1;
    for (Token token : tokens) {
      double value = idf(reader, token.term());
      if (value > bestIdf) {
        bestIdf = value;
        best = token.term();
      }
    }
    return best;
  }

  /** Adjacent bigram with the highest IDF sum that matches at least one document. */
  private static Query bestPhrase(IndexReader reader, IndexSearcher searcher, List<Token> tokens)
      throws IOException {
    PhraseQuery best = null;
    double bestValue = -1e9;
    for (int i = 0; i + 1 < tokens.size(); i++) {
      Token first = tokens.get(i);
      Token second = tokens.get(i + 1);
      if (first.term().equals(second.term())) {
        continue;
      }
      PhraseQuery candidate =
          new PhraseQuery.Builder()
              .add(term(first.term()), 0)
              .add(term(second.term()), second.position() - first.position())
              .build();
      double value = idf(reader, first.term()) + idf(reader, second.term());
      if (searcher.count(candidate) > 0 && value > bestValue) {
        bestValue = value;
        best = candidate;
      }
    }
    return best == null ? new MatchNoDocsQuery("no matching bigram") : best;
  }

  /** Builds each query class for one topic. Uses topic text and corpus statistics, never qrels. */
  private static Map<String, Query> buildQueries(
      IndexReader reader, IndexSearcher searcher, List<Token> tokens, LengthStats lengths)
      throws IOException {
    String rarest = rarestTerm(reader, tokens);
    String prefix = rarest.substring(0, Math.min(4, rarest.length()));
    String wildcard =
        rarest.length() >= 5
            ? rarest.substring(0, 3) + "*" + rarest.charAt(rarest.length() - 1)
            : rarest + "*";

    BooleanQuery.Builder bool = new BooleanQuery.Builder();
    bool.add(new TermQuery(term(rarest)), BooleanClause.Occur.MUST);
    for (Token token : tokens) {
      if (!token.term().equals(rarest)) {
        bool.add(new TermQuery(term(token.term())), BooleanClause.Occur.SHOULD);
      }
    }

    Map<String, Query> queries = new LinkedHashMap<>();
    queries.put("Term", new TermQuery(term(rarest)));
    queries.put("Phrase", bestPhrase(reader, searcher, tokens));
    queries.put("Range-Numeric", IntPoint.newRangeQuery(LENGTH_FIELD, lengths.q1(), lengths.q3()));
    queries.put("Range-Lexical", TermRangeQuery.newStringRange(FIELD, prefix, rarest, true, true));
    queries.put("Boolean", bool.build());
    queries.put("Prefix", new PrefixQuery(term(prefix)));
    queries.put("Wildcard", new WildcardQuery(term(wildcard)));
    queries.put("Fuzzy", new FuzzyQuery(term(rarest), 2));
    queries.put("MatchAll", new MatchAllDocsQuery());
    queries.put("OR-Baseline", orQuery(tokens)); // reference condition, not a ninth class
    return queries;
  }

  private static void runQueryTypeExperiment(
      IndexReader reader,
      IndexSearcher searcher,
      List<String[]> topics,
      LengthStats lengths,
      Path resultsDir)
      throws IOException {
    String[] types = {
      "Term", "Phrase", "Range-Numeric", "Range-Lexical", "Boolean",
      "Prefix", "Wildcard", "Fuzzy", "MatchAll", "OR-Baseline"
    };
    Map<String, PrintWriter> runFiles = new LinkedHashMap<>();
    for (String type : types) {
      runFiles.put(type, new PrintWriter(resultsDir.resolve("ss1_" + type + ".run").toFile()));
    }
    try (PrintWriter generated =
            new PrintWriter(resultsDir.resolve("ss1_generated_queries.tsv").toFile());
        PrintWriter counts = new PrintWriter(resultsDir.resolve("ss1_hit_counts.tsv").toFile())) {
      generated.println("qid\ttype\tquery");
      counts.println("qid\ttype\thits");
      for (String[] topic : topics) {
        int id = Integer.parseInt(topic[0]);
        List<Token> tokens = analyze(topic[2]);
        for (var entry : buildQueries(reader, searcher, tokens, lengths).entrySet()) {
          String type = entry.getKey();
          Query query = entry.getValue();
          writeRun(searcher, query, id, type, 100, runFiles.get(type));
          generated.println(id + "\t" + type + "\t" + query);
          counts.println(id + "\t" + type + "\t" + searcher.count(query));
        }
      }
    }
    runFiles.values().forEach(PrintWriter::close);
  }

  /** Hand-written queries showing how each class behaves, including word order and misspelling. */
  private static void runDemoQueries(
      IndexReader reader, IndexSearcher searcher, LengthStats lengths, Path resultsDir)
      throws IOException {
    Map<String, Query> demo = new LinkedHashMap<>();
    demo.put("TermQuery  boundari", new TermQuery(term("boundari")));
    demo.put("PhraseQuery  \"boundari layer\"", new PhraseQuery(FIELD, "boundari", "layer"));
    demo.put(
        "PhraseQuery  \"layer boundari\" (reversed)", new PhraseQuery(FIELD, "layer", "boundari"));
    demo.put(
        "PhraseQuery slop=2  \"layer boundari\"", new PhraseQuery(2, FIELD, "layer", "boundari"));
    demo.put(
        "TermRangeQuery  [bound TO boundz]",
        TermRangeQuery.newStringRange(FIELD, "bound", "boundz", true, true));
    demo.put(
        "IntPoint range  len in [" + lengths.q1() + "," + lengths.q3() + "]",
        IntPoint.newRangeQuery(LENGTH_FIELD, lengths.q1(), lengths.q3()));
    demo.put(
        "BooleanQuery  +boundari +layer -transit",
        new BooleanQuery.Builder()
            .add(new TermQuery(term("boundari")), BooleanClause.Occur.MUST)
            .add(new TermQuery(term("layer")), BooleanClause.Occur.MUST)
            .add(new TermQuery(term("transit")), BooleanClause.Occur.MUST_NOT)
            .build());
    demo.put("PrefixQuery  aero*", new PrefixQuery(term("aero")));
    demo.put("WildcardQuery  t?rbul*", new WildcardQuery(term("t?rbul*")));
    demo.put("FuzzyQuery  turbulanc~2 (misspelt)", new FuzzyQuery(term("turbulanc"), 2));
    demo.put("TermQuery  turbulanc (misspelt)", new TermQuery(term("turbulanc")));
    demo.put("MatchAllDocsQuery", new MatchAllDocsQuery());

    StoredFields fields = reader.storedFields();
    try (PrintWriter out = new PrintWriter(resultsDir.resolve("ss1_demo.tsv").toFile())) {
      out.println("query\thits\tdistinctTop100Scores\ttopScore\ttop5");
      for (var entry : demo.entrySet()) {
        ScoreDoc[] hits = searcher.search(entry.getValue(), 100).scoreDocs;
        Set<Float> distinctScores = new HashSet<>();
        for (ScoreDoc hit : hits) {
          distinctScores.add(hit.score);
        }
        StringBuilder top = new StringBuilder();
        for (int i = 0; i < Math.min(5, hits.length); i++) {
          top.append(i > 0 ? "," : "").append(fields.document(hits[i].doc).get("id"));
        }
        out.printf(
            Locale.ROOT,
            "%s\t%d\t%d\t%.3f\t%s%n",
            entry.getKey(),
            searcher.count(entry.getValue()),
            distinctScores.size(),
            hits.length > 0 ? hits[0].score : 0,
            top);
      }
    }
  }

  // ---------------------------------------------------------------- Part C

  /** Same index, analyzer and queries for every model; only the scoring function changes. */
  private static void runRetrievalModels(
      IndexReader reader, List<String[]> topics, Path resultsDir) throws IOException {
    Map<String, Similarity> models = new LinkedHashMap<>();
    models.put("TF-IDF", new ClassicSimilarity());
    models.put("BM25", new BM25Similarity(1.2f, 0.75f));
    models.put("LM-Dirichlet", new LMDirichletSimilarity(2000f));
    // Sensitivity variants, reported separately from the default-parameter comparison.
    for (float mu : new float[] {50f, 100f, 200f, 500f, 1000f}) {
      models.put("LMD-mu" + (int) mu, new LMDirichletSimilarity(mu));
    }
    for (float lambda : new float[] {0.1f, 0.3f, 0.7f}) {
      models.put("LMJM-l" + lambda, new LMJelinekMercerSimilarity(lambda));
    }
    for (float b : new float[] {0.25f, 0.5f, 1.0f}) {
      models.put("BM25-b" + b, new BM25Similarity(1.2f, b));
    }

    for (var model : models.entrySet()) {
      IndexSearcher searcher = new IndexSearcher(reader);
      searcher.setSimilarity(model.getValue());
      Path file = resultsDir.resolve("partc_" + model.getKey() + ".run");
      try (PrintWriter out = new PrintWriter(file.toFile())) {
        for (String[] topic : topics) {
          Query query = orQuery(analyze(topic[2]));
          writeRun(searcher, query, Integer.parseInt(topic[0]), model.getKey(), 1000, out);
        }
      }
    }
  }
}
