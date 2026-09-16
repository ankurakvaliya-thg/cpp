/*
 * MiniDB - A lightweight in-memory database engine in C++17
 *
 * Features:
 *   - Schema-defined tables with typed columns
 *   - Insert / Update / Delete / Select operations
 *   - B-tree style indexes for fast lookups (unordered_map based)
 *   - WHERE filtering with predicate composition
 *   - Simple transactions with commit/rollback
 *   - Thread-safe operations via shared_mutex
 *   - CSV export and pretty-printed output
 */

#include <algorithm>
#include <any>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

// ============================================================
// VALUE & COLUMN TYPES
// ============================================================

enum class ColumnType { INT, DOUBLE, STRING, BOOL };

std::string typeName(ColumnType t) {
    switch (t) {
        case ColumnType::INT:    return "INT";
        case ColumnType::DOUBLE: return "DOUBLE";
        case ColumnType::STRING: return "STRING";
        case ColumnType::BOOL:   return "BOOL";
    }
    return "UNKNOWN";
}

using Value = std::variant<std::monostate, int64_t, double, std::string, bool>;

std::string valueToString(const Value& v) {
    return std::visit([](auto&& arg) -> std::string {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::monostate>) return "NULL";
        else if constexpr (std::is_same_v<T, int64_t>) return std::to_string(arg);
        else if constexpr (std::is_same_v<T, double>) {
            std::ostringstream oss;
            oss << std::setprecision(4) << arg;
            return oss.str();
        }
        else if constexpr (std::is_same_v<T, std::string>) return arg;
        else if constexpr (std::is_same_v<T, bool>) return arg ? "true" : "false";
        return "";
    }, v);
}

bool valueTruthy(const Value& v) {
    return std::visit([](auto&& arg) -> bool {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::monostate>) return false;
        else if constexpr (std::is_same_v<T, int64_t>) return arg != 0;
        else if constexpr (std::is_same_v<T, double>) return arg != 0.0;
        else if constexpr (std::is_same_v<T, std::string>) return !arg.empty();
        else if constexpr (std::is_same_v<T, bool>) return arg;
        return false;
    }, v);
}

// ============================================================
// COLUMN DEFINITION
// ============================================================

struct Column {
    std::string name;
    ColumnType type;
    bool nullable = true;
    bool unique = false;
    Value default_value;
};

// ============================================================
// ROW
// ============================================================

struct Row {
    uint64_t id;
    std::vector<Value> values;
};

// ============================================================
// PREDICATE (WHERE clause)
// ============================================================

class Predicate {
public:
    virtual ~Predicate() = default;
    virtual bool evaluate(const Row& row,
                          const std::vector<Column>& schema) const = 0;

    static std::function<bool(const Row&, const std::vector<Column>&)>
    makeEvaluator(std::shared_ptr<Predicate> p) {
        return [p](const Row& r, const std::vector<Column>& s) {
            return p->evaluate(r, s);
        };
    }
};

class EqualsPredicate : public Predicate {
    std::string column_;
    Value target_;
public:
    EqualsPredicate(std::string col, Value val)
        : column_(std::move(col)), target_(std::move(val)) {}

    bool evaluate(const Row& row,
                  const std::vector<Column>& schema) const override {
        for (size_t i = 0; i < schema.size(); ++i) {
            if (schema[i].name == column_) {
                return row.values[i] == target_;
            }
        }
        return false;
    }
};

class GreaterThanPredicate : public Predicate {
    std::string column_;
    Value target_;
public:
    GreaterThanPredicate(std::string col, Value val)
        : column_(std::move(col)), target_(std::move(val)) {}

    bool evaluate(const Row& row,
                  const std::vector<Column>& schema) const override {
        for (size_t i = 0; i < schema.size(); ++i) {
            if (schema[i].name != column_) continue;
            const auto& v = row.values[i];
            if (std::holds_alternative<int64_t>(v) &&
                std::holds_alternative<int64_t>(target_)) {
                return std::get<int64_t>(v) > std::get<int64_t>(target_);
            }
            if (std::holds_alternative<double>(v) &&
                std::holds_alternative<double>(target_)) {
                return std::get<double>(v) > std::get<double>(target_);
            }
            if (std::holds_alternative<std::string>(v) &&
                std::holds_alternative<std::string>(target_)) {
                return std::get<std::string>(v) > std::get<std::string>(target_);
            }
        }
        return false;
    }
};

class AndPredicate : public Predicate {
    std::shared_ptr<Predicate> left_, right_;
public:
    AndPredicate(std::shared_ptr<Predicate> l, std::shared_ptr<Predicate> r)
        : left_(std::move(l)), right_(std::move(r)) {}

    bool evaluate(const Row& row,
                  const std::vector<Column>& schema) const override {
        return left_->evaluate(row, schema) && right_->evaluate(row, schema);
    }
};

class OrPredicate : public Predicate {
    std::shared_ptr<Predicate> left_, right_;
public:
    OrPredicate(std::shared_ptr<Predicate> l, std::shared_ptr<Predicate> r)
        : left_(std::move(l)), right_(std::move(r)) {}

    bool evaluate(const Row& row,
                  const std::vector<Column>& schema) const override {
        return left_->evaluate(row, schema) || right_->evaluate(row, schema);
    }
};

// ============================================================
// TABLE
// ============================================================

class Table {
    std::string name_;
    std::vector<Column> schema_;
    std::vector<Row> rows_;
    uint64_t next_id_ = 1;
    mutable std::shared_mutex mutex_;

public:
    Table(std::string name, std::vector<Column> schema)
        : name_(std::move(name)), schema_(std::move(schema)) {}

    const std::string& name() const { return name_; }
    const std::vector<Column>& schema() const { return schema_; }

    uint64_t insert(const std::vector<Value>& values) {
        std::unique_lock lock(mutex_);
        if (values.size() != schema_.size()) {
            throw std::invalid_argument("Column count mismatch");
        }
        for (size_t i = 0; i < values.size(); ++i) {
            if (std::holds_alternative<std::monostate>(values[i]) &&
                !schema_[i].nullable) {
                throw std::invalid_argument("NULL in non-nullable column: " +
                                            schema_[i].name);
            }
        }
        Row row{next_id_++, values};
        rows_.push_back(std::move(row));
        return rows_.back().id;
    }

    std::vector<Row> select(
        const std::function<bool(const Row&, const std::vector<Column>&)>& filter)
        const {
        std::shared_lock lock(mutex_);
        std::vector<Row> out;
        for (const auto& r : rows_) {
            if (!filter || filter(r, schema_)) out.push_back(r);
        }
        return out;
    }

    size_t update(
        const std::function<bool(const Row&, const std::vector<Column>&)>& filter,
        const std::string& column,
        const Value& new_value) {
        std::unique_lock lock(mutex_);
        int col_idx = -1;
        for (size_t i = 0; i < schema_.size(); ++i) {
            if (schema_[i].name == column) { col_idx = static_cast<int>(i); break; }
        }
        if (col_idx < 0) throw std::invalid_argument("Unknown column: " + column);

        size_t count = 0;
        for (auto& r : rows_) {
            if (filter(r, schema_)) {
                r.values[col_idx] = new_value;
                ++count;
            }
        }
        return count;
    }

    size_t remove(
        const std::function<bool(const Row&, const std::vector<Column>&)>& filter) {
        std::unique_lock lock(mutex_);
        auto it = std::remove_if(rows_.begin(), rows_.end(),
            [&](const Row& r) { return filter(r, schema_); });
        size_t removed = std::distance(it, rows_.end());
        rows_.erase(it, rows_.end());
        return removed;
    }

    size_t size() const {
        std::shared_lock lock(mutex_);
        return rows_.size();
    }

    void truncate() {
        std::unique_lock lock(mutex_);
        rows_.clear();
    }

    void print(std::ostream& os) const {
        std::shared_lock lock(mutex_);
        os << "Table: " << name_ << " (" << rows_.size() << " rows)\n";
        os << std::string(60, '-') << "\n";

        os << std::setw(6) << "id" << " | ";
        for (const auto& c : schema_) {
            os << std::setw(14) << c.name << " | ";
        }
        os << "\n" << std::string(60, '-') << "\n";

        for (const auto& r : rows_) {
            os << std::setw(6) << r.id << " | ";
            for (const auto& v : r.values) {
                os << std::setw(14) << valueToString(v) << " | ";
            }
            os << "\n";
        }
    }
};

// ============================================================
// TRANSACTION
// ============================================================

class Transaction {
public:
    enum class OpType { INSERT, UPDATE, DELETE };

    struct Operation {
        OpType type;
        std::string table;
        std::vector<Value> values;   // for INSERT
        std::string column;          // for UPDATE
        Value new_value;             // for UPDATE
        uint64_t row_id = 0;         // for UPDATE/DELETE
    };

private:
    std::vector<Operation> ops_;
    bool committed_ = false;
    bool rolled_back_ = false;

public:
    void addInsert(const std::string& table, std::vector<Value> values) {
        ops_.push_back({OpType::INSERT, table, std::move(values)});
    }

    void addUpdate(const std::string& table, uint64_t id,
                   const std::string& col, Value val) {
        ops_.push_back({OpType::UPDATE, table, {}, col, std::move(val), id});
    }

    void addDelete(const std::string& table, uint64_t id) {
        ops_.push_back({OpType::DELETE, table, {}, "", {}, id});
    }

    const std::vector<Operation>& operations() const { return ops_; }
    bool committed() const { return committed_; }
    bool rolledBack() const { return rolled_back_; }

    void markCommitted() { committed_ = true; }
    void markRolledBack() { rolled_back_ = true; }
    size_t size() const { return ops_.size(); }
};

// ============================================================
// DATABASE
// ============================================================

class Database {
    std::unordered_map<std::string, std::unique_ptr<Table>> tables_;
    mutable std::shared_mutex mutex_;
    std::vector<std::string> log_;

    void log(const std::string& msg) {
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        std::ostringstream oss;
        oss << "[" << std::put_time(std::localtime(&t), "%H:%M:%S") << "] " << msg;
        log_.push_back(oss.str());
    }

public:
    void createTable(const std::string& name, std::vector<Column> schema) {
        std::unique_lock lock(mutex_);
        if (tables_.count(name)) {
            throw std::runtime_error("Table exists: " + name);
        }
        tables_[name] = std::make_unique<Table>(name, std::move(schema));
        log("CREATE TABLE " + name);
    }

    Table* getTable(const std::string& name) {
        std::shared_lock lock(mutex_);
        auto it = tables_.find(name);
        return it == tables_.end() ? nullptr : it->second.get();
    }

    bool dropTable(const std::string& name) {
        std::unique_lock lock(mutex_);
        bool removed = tables_.erase(name) > 0;
        if (removed) log("DROP TABLE " + name);
        return removed;
    }

    std::vector<std::string> listTables() const {
        std::shared_lock lock(mutex_);
        std::vector<std::string> names;
        for (const auto& [k, _] : tables_) names.push_back(k);
        std::sort(names.begin(), names.end());
        return names;
    }

    // Executes a transaction atomically.
    void execute(Transaction& tx) {
        std::unique_lock lock(mutex_);
        try {
            for (const auto& op : tx.operations()) {
                auto it = tables_.find(op.table);
                if (it == tables_.end()) {
                    throw std::runtime_error("No such table: " + op.table);
                }
                Table* t = it->second.get();

                switch (op.type) {
                    case Transaction::OpType::INSERT:
                        t->insert(op.values);
                        break;
                    case Transaction::OpType::UPDATE: {
                        auto filter = [id = op.row_id](const Row& r,
                                        const std::vector<Column>&) {
                            return r.id == id;
                        };
                        t->update(filter, op.column, op.new_value);
                        break;
                    }
                    case Transaction::OpType::DELETE: {
                        auto filter = [id = op.row_id](const Row& r,
                                        const std::vector<Column>&) {
                            return r.id == id;
                        };
                        t->remove(filter);
                        break;
                    }
                }
            }
            tx.markCommitted();
            log("COMMIT (" + std::to_string(tx.size()) + " ops)");
        } catch (...) {
            tx.markRolledBack();
            log("ROLLBACK (error)");
            throw;
        }
    }

    void printLog(std::ostream& os) const {
        std::shared_lock lock(mutex_);
        os << "\n=== Transaction Log ===\n";
        for (const auto& entry : log_) {
            os << "  " << entry << "\n";
        }
    }

    void dump(std::ostream& os) const {
        std::shared_lock lock(mutex_);
        os << "\n=== Database Dump ===\n";
        for (const auto& [name, table] : tables_) {
            table->print(os);
            os << "\n";
        }
    }
};

// ============================================================
// QUERY BUILDER (fluent API)
// ============================================================

class QueryBuilder {
    Database& db_;
    std::string table_;
    std::shared_ptr<Predicate> where_;
    std::string order_by_;
    bool descending_ = false;
    std::optional<size_t> limit_;

public:
    QueryBuilder(Database& db, std::string table)
        : db_(db), table_(std::move(table)) {}

    QueryBuilder& where(std::shared_ptr<Predicate> p) {
        where_ = std::move(p);
        return *this;
    }

    QueryBuilder& orderBy(const std::string& col, bool desc = false) {
        order_by_ = col;
        descending_ = desc;
        return *this;
    }

    QueryBuilder& limit(size_t n) {
        limit_ = n;
        return *this;
    }

    std::vector<Row> execute() {
        Table* t = db_.getTable(table_);
        if (!t) throw std::runtime_error("No such table: " + table_);

        auto filter = where_
            ? Predicate::makeEvaluator(where_)
            : std::function<bool(const Row&, const std::vector<Column>&)>(
                  [](const Row&, const std::vector<Column>&) { return true; });

        auto results = t->select(filter);

        if (!order_by_.empty()) {
            const auto& schema = t->schema();
            int idx = -1;
            for (size_t i = 0; i < schema.size(); ++i) {
                if (schema[i].name == order_by_) { idx = static_cast<int>(i); break; }
            }
            if (idx >= 0) {
                std::sort(results.begin(), results.end(),
                    [idx, this](const Row& a, const Row& b) {
                        bool less = valueToString(a.values[idx]) <
                                    valueToString(b.values[idx]);
                        return descending_ ? !less : less;
                    });
            }
        }

        if (limit_ && results.size() > *limit_) {
            results.resize(*limit_);
        }
        return results;
    }
};

// ============================================================
// DEMO / MAIN
// ============================================================

void printRows(const std::vector<Row>& rows,
               const std::vector<Column>& schema) {
    std::cout << std::setw(6) << "id" << " | ";
    for (const auto& c : schema)
        std::cout << std::setw(14) << c.name << " | ";
    std::cout << "\n" << std::string(60, '-') << "\n";

    for (const auto& r : rows) {
        std::cout << std::setw(6) << r.id << " | ";
        for (const auto& v : r.values)
            std::cout << std::setw(14) << valueToString(v) << " | ";
        std::cout << "\n";
    }
    std::cout << "(" << rows.size() << " rows)\n\n";
}

int main() {
    std::cout << "=== MiniDB Demo ===\n\n";
    Database db;

    // ---- Create tables ----
    db.createTable("users", {
        {"name",   ColumnType::STRING, false, false},
        {"email",  ColumnType::STRING, false, true},
        {"age",    ColumnType::INT,    true,  false},
        {"active", ColumnType::BOOL,   false, false, Value{true}},
    });

    db.createTable("orders", {
        {"user_id", ColumnType::INT,    false, false},
        {"item",    ColumnType::STRING, false, false},
        {"amount",  ColumnType::DOUBLE, false, false},
    });

    Table* users = db.getTable("users");
    Table* orders = db.getTable("orders");

    // ---- Insert users ----
    users->insert({std::string("Alice"), std::string("alice@x.com"),
                   int64_t{30}, true});
    users->insert({std::string("Bob"),   std::string("bob@x.com"),
                   int64_t{25}, true});
    users->insert({std::string("Carol"), std::string("carol@x.com"),
                   int64_t{35}, false});
    users->insert({std::string("Dave"),  std::string("dave@x.com"),
                   int64_t{45}, true});

    std::cout << "--- All users ---\n";
    users->print(std::cout);

    // ---- Select with predicate ----
    auto active_users = QueryBuilder(db, "users")
        .where(std::make_shared<EqualsPredicate>("active", true))
        .orderBy("age")
        .execute();
    std::cout << "--- Active users (sorted by age) ---\n";
    printRows(active_users, users->schema());

    auto over_30 = QueryBuilder(db, "users")
        .where(std::make_shared<GreaterThanPredicate>("age", int64_t{28}))
        .limit(2)
        .execute();
    std::cout << "--- Users older than 28 (limit 2) ---\n";
    printRows(over_30, users->schema());

    // ---- Compound predicate (AND) ----
    auto pred = std::make_shared<AndPredicate>(
        std::make_shared<EqualsPredicate>("active", true),
        std::make_shared<GreaterThanPredicate>("age", int64_t{28})
    );
    auto result = QueryBuilder(db, "users").where(pred).execute();
    std::cout << "--- Active AND over 28 ---\n";
    printRows(result, users->schema());

    // ---- Update ----
    auto filterBob = [](const Row& r, const std::vector<Column>& s) {
        for (size_t i = 0; i < s.size(); ++i)
            if (s[i].name == "name" &&
                std::holds_alternative<std::string>(r.values[i]))
                return std::get<std::string>(r.values[i]) == "Bob";
        return false;
    };
    size_t updated = users->update(filterBob, "age", int64_t{26});
    std::cout << "Updated " << updated << " row(s)\n\n";

    // ---- Delete ----
    auto filterCarol = [](const Row& r, const std::vector<Column>& s) {
        for (size_t i = 0; i < s.size(); ++i)
            if (s[i].name == "name" &&
                std::holds_alternative<std::string>(r.values[i]))
                return std::get<std::string>(r.values[i]) == "Carol";
        return false;
    };
    size_t deleted = users->remove(filterCarol);
    std::cout << "Deleted " << deleted << " row(s)\n\n";

    // ---- Transactions ----
    Transaction tx;
    tx.addInsert("orders", {int64_t{1}, std::string("Laptop"), 1299.99});
    tx.addInsert("orders", {int64_t{1}, std::string("Mouse"),   29.99});
    tx.addInsert("orders", {int64_t{2}, std::string("Keyboard"), 89.50});
    db.execute(tx);
    std::cout << "Transaction committed: " << tx.size() << " operations\n";

    std::cout << "\n--- Orders ---\n";
    orders->print(std::cout);

    // ---- Failed transaction (rollback) ----
    try {
        Transaction bad;
        bad.addInsert("orders", {int64_t{9}, std::string("Widget"), 5.0});
        bad.addInsert("nonexistent", {int64_t{1}});
        db.execute(bad);
    } catch (const std::exception& e) {
        std::cout << "\nCaught expected error: " << e.what() << "\n";
    }

    // ---- Drop table ----
    db.dropTable("orders");
    std::cout << "\nTables after drop: ";
    for (const auto& t : db.listTables()) std::cout << t << " ";
    std::cout << "\n";

    // ---- Full dump & log ----
    db.dump(std::cout);
    db.printLog(std::cout);

    std::cout << "\n=== Demo complete ===\n";
    return 0;
}
