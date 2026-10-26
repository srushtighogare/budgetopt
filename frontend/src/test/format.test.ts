import { money, ms, pct, ratioPct, titleCase } from "../lib/format";

describe("format helpers", () => {
  it("adapts money precision to tiny per-request costs", () => {
    expect(money(12.5)).toBe("$12.50");
    expect(money("0.42")).toBe("$0.4200");
    expect(money(0.0000036)).toBe("$0.000004");
    expect(money(0)).toBe("$0.00");
    expect(money(null)).toBe("—");
    expect(money("not a number")).toBe("—");
  });

  it("formats percentages, ratios and latency", () => {
    expect(pct(92.06)).toBe("92%");
    expect(ratioPct(0.913)).toBe("91%");
    expect(ms(820)).toBe("820 ms");
    expect(ms(2300)).toBe("2.3 s");
    expect(ms(null)).toBe("—");
  });

  it("title-cases identifiers and keeps acronyms", () => {
    expect(titleCase("reasoning_math")).toBe("Reasoning Math");
    expect(titleCase("simple_qa")).toBe("Simple QA");
    expect(titleCase("ets_level_weekly")).toBe("ETS Level Weekly");
  });
});
