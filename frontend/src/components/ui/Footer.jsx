import { Link } from 'react-router-dom';
import { Github, Linkedin, Instagram, GraduationCap } from 'lucide-react';

const socials = [
  { href: 'https://www.instagram.com/3._.sree/', label: 'Instagram', Icon: Instagram },
  { href: 'https://github.com/Sreevatsan03hub', label: 'GitHub', Icon: Github },
  { href: 'https://www.linkedin.com/in/sreevatsan-v-p-866956294/', label: 'LinkedIn', Icon: Linkedin },
];

const footerLinks = {
  Platform: [{ to: '/', label: 'Home' }, { to: '/dashboard', label: 'Dashboard' }, { to: '/history', label: 'History' }],
  Resources: [{ to: '/settings', label: 'Settings' }, { to: '#', label: 'Help & Support' }, { to: '#', label: 'Privacy Policy' }],
};

export function Footer() {
  const year = new Date().getFullYear();
  return (
    <footer className="bg-bg-dark text-blue-200">
      <div className="max-w-7xl mx-auto px-6 pt-14 pb-8">

        {/* ── Top grid ─────────────────────── */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-10 mb-12">
          {/* Brand */}
          <div className="md:col-span-2">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
                <GraduationCap size={17} className="text-white" />
              </div>
              <span className="text-white text-xl font-bold">Accessify</span>
            </div>
            <p className="text-sm text-blue-300 leading-relaxed max-w-xs">
              AI-powered real-time captions and multilingual translation making
              education accessible for everyone — everywhere.
            </p>
            {/* Social icons */}
            <div className="flex items-center gap-4 mt-6">
              {socials.map(({ href, label, Icon }) => (
                <a
                  key={label}
                  href={href}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label={label}
                  className="w-9 h-9 rounded-lg bg-white/10 flex items-center justify-center
                             text-blue-300 hover:bg-primary hover:text-white transition-all"
                >
                  <Icon size={16} />
                </a>
              ))}
            </div>
          </div>

          {/* Link columns */}
          {Object.entries(footerLinks).map(([heading, links]) => (
            <div key={heading}>
              <h4 className="text-white text-sm font-bold mb-4 tracking-wider uppercase">{heading}</h4>
              <ul className="space-y-2.5">
                {links.map(({ to, label }) => (
                  <li key={label}>
                    <Link
                      to={to}
                      className="text-sm text-blue-300 hover:text-white transition-colors"
                    >
                      {label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* ── Bottom bar ───────────────────── */}
        <div className="border-t border-white/10 pt-6 flex flex-col sm:flex-row
                        justify-between items-center gap-3">
          <p className="text-blue-400 text-xs">
            © {year} Accessify. All rights reserved.
          </p>
          <div className="flex items-center gap-5">
            {socials.map(({ href, label, Icon }) => (
              <a
                key={label}
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 text-xs text-blue-400 hover:text-white transition-colors"
              >
                <Icon size={13} />
                {label}
              </a>
            ))}
          </div>
        </div>

      </div>
    </footer>
  );
}
