import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { OVERLAY_FIXTURE } from "@/lib/overlay-fixture";
import { RepFilmstrip } from "@/components/rep-filmstrip";

const REPS = OVERLAY_FIXTURE.reps;

describe("RepFilmstrip", () => {
  it("renders one card per rep with score, verdict, and unanalyzed reason", () => {
    render(<RepFilmstrip reps={REPS} selected={0} onSelect={() => {}} />);

    const cards = screen.getAllByTestId("rep-card");
    expect(cards).toHaveLength(5);

    // Made rep shows its score in the accent style.
    expect(screen.getByText("84")).toBeInTheDocument();
    expect(screen.getByText("91")).toBeInTheDocument();
    expect(screen.getAllByText("Made")).toHaveLength(3);

    // Missed + unanalyzed reps show an em dash for the score.
    expect(screen.getAllByText("—")).toHaveLength(2);
    expect(screen.getByText("Missed")).toBeInTheDocument();

    // Unanalyzed rep surfaces its reason.
    expect(screen.getByText("Unanalyzed")).toBeInTheDocument();
    expect(
      screen.getByText("bar marker lost during catch"),
    ).toBeInTheDocument();
  });

  it("marks the selected card via aria-pressed", () => {
    render(<RepFilmstrip reps={REPS} selected={2} onSelect={() => {}} />);
    const cards = screen.getAllByTestId("rep-card");
    expect(cards[2]).toHaveAttribute("aria-pressed", "true");
    expect(cards[0]).toHaveAttribute("aria-pressed", "false");
  });

  it("clicking a card selects that rep's index", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(<RepFilmstrip reps={REPS} selected={0} onSelect={onSelect} />);

    await user.click(screen.getAllByTestId("rep-card")[3]);
    expect(onSelect).toHaveBeenCalledWith(3);
  });

  it("prev/next step the selection and disable at the ends", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    const { rerender } = render(
      <RepFilmstrip reps={REPS} selected={0} onSelect={onSelect} />,
    );

    expect(screen.getByRole("button", { name: /previous rep/i })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: /next rep/i }));
    expect(onSelect).toHaveBeenCalledWith(1);

    rerender(<RepFilmstrip reps={REPS} selected={4} onSelect={onSelect} />);
    expect(screen.getByRole("button", { name: /next rep/i })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: /previous rep/i }));
    expect(onSelect).toHaveBeenCalledWith(3);
  });
});
