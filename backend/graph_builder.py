import pandas as pd
from neo4j import GraphDatabase

# Neo4j connection
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4j123"

CSV_FILE = "../data/processed_tickets.csv"

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

def run_query(query, parameters=None):
    with driver.session() as session:
        session.run(query, parameters)

print("Clearing old data...")
run_query("MATCH (n) DETACH DELETE n")

print("Loading CSV...")
df = pd.read_csv(CSV_FILE)

for _, row in df.iterrows():

    run_query("""
    MERGE (c:Customer {
        email: $email
    })
    SET c.name = $name,
        c.age = $age,
        c.gender = $gender
    """, {
        "email": row["customer_email"],
        "name": row["customer_name"],
        "age": int(row["customer_age"]),
        "gender": row["customer_gender"]
    })

    run_query("""
    MERGE (p:Product {
        product_name: $product
    })
    """, {
        "product": row["product_purchased"]
    })

    run_query("""
    MERGE (t:Ticket {
        ticket_id: $ticket_id
    })
    SET t.subject = $subject,
        t.status = $status
    """, {
        "ticket_id": row["ticket_id"],
        "subject": row["ticket_subject"],
        "status": row["ticket_status"]
    })

    run_query("""
    MERGE (i:Issue {
        issue_name: $issue
    })
    """, {
        "issue": row["ticket_subject"]
    })

    # Relationships
    run_query("""
    MATCH (c:Customer {email: $email})
    MATCH (t:Ticket {ticket_id: $ticket_id})
    MERGE (c)-[:RAISED]->(t)
    """, {
        "email": row["customer_email"],
        "ticket_id": row["ticket_id"]
    })

    run_query("""
    MATCH (t:Ticket {ticket_id: $ticket_id})
    MATCH (p:Product {product_name: $product})
    MERGE (t)-[:ABOUT]->(p)
    """, {
        "ticket_id": row["ticket_id"],
        "product": row["product_purchased"]
    })

    run_query("""
    MATCH (t:Ticket {ticket_id: $ticket_id})
    MATCH (i:Issue {issue_name: $issue})
    MERGE (t)-[:HAS_ISSUE]->(i)
    """, {
        "ticket_id": row["ticket_id"],
        "issue": row["ticket_subject"]
    })

    run_query("""
    MATCH (c:Customer {email: $email})
    MATCH (p:Product {product_name: $product})
    MERGE (c)-[:BOUGHT]->(p)
    """, {
        "email": row["customer_email"],
        "product": row["product_purchased"]
    })

print("✅ Graph Created Successfully!")
driver.close()