import { createClient } from "@supabase/supabase-js";

/**
 * Service-role Supabase client for server-side operations.
 * Uses the service role key which bypasses Row Level Security.
 * NEVER expose this client or key to the browser.
 */
const supabaseAdmin = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY,
);

export default supabaseAdmin;
