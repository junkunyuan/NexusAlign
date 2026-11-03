---
paths:
  - "**/*"
---

# Writing

Write all comments, docstrings, config comments, and docs in ASD-STE100 Simplified
Technical English (STE). STE makes the text easy to read, also for readers who are not
native English speakers.

1. **Use simple words with one meaning.** For example, use "use", not "utilize";
   "get", not "obtain"; "make sure", not "ensure"; "with" or "through", not "via".
   Use one word for one thing: always "checkpoint", never "ckpt" or "snapshot".
2. **Keep technical names as they are,** such as FSDP, EMA, LoRA, and timestep, and
   names from the code, such as `cfg_omega`. Use an "-ing" word only as a technical
   name, such as "training" or "sampling".
3. **Do not use more than three words in a noun cluster.** Write "the CFG weight of
   the mean velocity", not "mean velocity CFG weight".
4. **Use the active voice and simple tenses.** Write "the trainer saves a checkpoint",
   not "a checkpoint is saved".
5. **Write one topic per sentence.** Use not more than 20 words in an instruction and
   25 words in a description. Use a vertical list for complex text.
6. **Write instructions in the imperative,** one per sentence, with the condition
   first: "If the run stops, start it again."
7. **Do not omit words to make a sentence shorter.** Keep "a", "an", and "the". A
   trailing comment that only names a value, such as `# EMA decay`, can be a noun
   phrase.
8. **Do not use contractions, Latin abbreviations, or idioms.** Write "do not", not
   "don't"; "for example", not "e.g."; "that is", not "i.e.".
9. **Write one topic per paragraph,** with not more than six sentences. Start with
   the topic sentence.
