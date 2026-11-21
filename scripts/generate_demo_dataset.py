from __future__ import annotations

import csv
import random
from pathlib import Path

from faker import Faker


SEED = 42
N_ROWS = 250

OUTPUT_PATH = Path("demo_data/privacyguard_ai_demo_synthetic_sensitive_dataset.csv")

fake = Faker("en_US")
Faker.seed(SEED)
random.seed(SEED)


DEPARTMENTS = [
    "Sales",
    "Marketing",
    "Finance",
    "HR",
    "Operations",
    "Customer Support",
    "Product",
    "IT",
    "Legal",
    "Business Development",
]

JOB_TITLES = [
    "Sales Representative",
    "Account Manager",
    "Marketing Specialist",
    "Financial Analyst",
    "HR Coordinator",
    "Operations Manager",
    "Customer Support Agent",
    "Product Analyst",
    "Software Developer",
    "Legal Assistant",
    "Data Analyst",
    "Project Coordinator",
]

MEDICAL_CONDITIONS = [
    "None",
    "Asthma",
    "Diabetes",
    "Hypertension",
    "Migraine",
    "Anxiety",
    "Arthritis",
]

CITIES_AND_POSTAL_CODES = {
    "Krakow": ["30-001", "30-002", "31-001", "31-042"],
    "Warsaw": ["00-001", "00-002", "01-234", "02-495"],
    "Wroclaw": ["50-001", "50-002", "51-100", "53-611"],
    "Gdansk": ["80-001", "80-002", "80-180", "80-462"],
    "Poznan": ["60-001", "60-002", "61-001", "61-772"],
    "Katowice": ["40-001", "40-002", "40-098", "40-750"],
    "Lodz": ["90-001", "90-002", "91-001", "93-578"],
    "Lublin": ["20-001", "20-002", "20-400", "20-950"],
}

NOTE_TEMPLATES = [
    "Customer {name} requested phone contact at {phone}.",
    "Contact {name} by email: {email}.",
    "Customer lives near {city} and prefers morning calls.",
    "Employee reported {condition} during onboarding.",
    "Customer {name} asked to update address to {address}.",
    "Follow up with {name}; alternate phone number is {phone}.",
    "No special notes.",
    "General customer service note.",
    "Billing question resolved.",
    "Customer requested newsletter opt-out.",
]


def generate_customer_id(index: int) -> str:
    return f"CUST-{index:04d}"


def generate_phone_number() -> str:
    return f"+48 {random.randint(500, 799)} {random.randint(100, 999)} {random.randint(100, 999)}"


def generate_row(index: int) -> dict[str, object]:
    first_name = fake.first_name()
    last_name = fake.last_name()
    full_name = f"{first_name} {last_name}"

    email = f"{first_name.lower()}.{last_name.lower()}{index}@example.com"
    phone_number = generate_phone_number()
    address = fake.street_address()

    city = random.choice(list(CITIES_AND_POSTAL_CODES.keys()))
    postal_code = random.choice(CITIES_AND_POSTAL_CODES[city])

    age = random.randint(18, 78)
    gender = random.choice(["Female", "Male", "Other"])

    department = random.choice(DEPARTMENTS)
    job_title = random.choice(JOB_TITLES)

    salary = round(random.uniform(38_000, 165_000), 2)
    medical_condition = random.choices(
        MEDICAL_CONDITIONS,
        weights=[35, 12, 12, 12, 10, 10, 9],
        k=1,
    )[0]

    purchase_amount = round(random.uniform(15, 2_500), 2)
    loyalty_score = round(random.uniform(0, 100), 2)
    signup_date = fake.date_between(start_date="-7y", end_date="today").isoformat()

    note_template = random.choice(NOTE_TEMPLATES)
    notes = note_template.format(
        name=full_name,
        phone=phone_number,
        email=email,
        city=city,
        condition=medical_condition,
        address=address,
    )

    return {
        "customer_id": generate_customer_id(index),
        "full_name": full_name,
        "email": email,
        "phone_number": phone_number,
        "address": address,
        "city": city,
        "postal_code": postal_code,
        "age": age,
        "gender": gender,
        "department": department,
        "job_title": job_title,
        "salary": salary,
        "medical_condition": medical_condition,
        "purchase_amount": purchase_amount,
        "loyalty_score": loyalty_score,
        "signup_date": signup_date,
        "notes": notes,
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = [generate_row(index) for index in range(1, N_ROWS + 1)]

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} rows")
    print(f"Saved dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()